import functools
from typing import Tuple, Callable, Any, List, Optional, Union, Dict, Mapping

import scenic.train_lib.train_utils as train_utils

from absl import logging
import flax
import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.base_models.multilabel_classification_model import (
    MultiLabelClassificationModel,
)
from scenic.model_lib.layers import nn_layers

from scenic.projects.glc.adapters import BottleneckAdapterParallel
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.profiler
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
from scenic.projects.glc  import adapters as adapters
import pickle

PyTree = Union[Mapping[str, Mapping], Any]



def ids_str2ints(ids_str):
    return [int(v) for v in ids_str.split("_")] if ids_str else []


def ids_ints2str(ids_ints):
    return "_".join([str(v) for v in sorted(ids_ints)])

def _replace_dict(
    model: PyTree,
    restored: PyTree,
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None,
) -> PyTree:
    """Replaces values in model dictionary with restored ones from checkpoint."""
    # import pdb; pdb.set_trace()
    name_mapping = name_mapping or {}

    model = flax.core.unfreeze(model)  # pytype: disable=wrong-arg-types
    restored = flax.core.unfreeze(restored)  # pytype: disable=wrong-arg-types

    if ckpt_prefix_path:
        for p in ckpt_prefix_path:
            restored = restored[p]

    if model_prefix_path:
        for p in reversed(model_prefix_path):
            restored = {p: restored}

    # Flatten nested parameters to a dict of str -> tensor. Keys are tuples
    # from the path in the nested dictionary to the specific tensor. E.g.,
    # {'a1': {'b1': t1, 'b2': t2}, 'a2': t3}
    # -> {('a1', 'b1'): t1, ('a1', 'b2'): t2, ('a2',): t3}.
    restored_flat = flax.traverse_util.flatten_dict(
        dict(restored), keep_empty_nodes=True
    )
    model_flat = flax.traverse_util.flatten_dict(dict(model), keep_empty_nodes=True)
    # import pdb;pdb.set_trace()
    # ('/init_bn').strip("/").split("/")
    for m_key, m_params in restored_flat.items():

        # pytype: disable=attribute-error
        for name, to_replace in name_mapping.items():
            m_key = tuple(to_replace if k == name else k for k in m_key)
        # pytype: enable=attribute-error
        par = []
        for elem in m_key:
            elem = elem.strip("/").split("/")
            par += elem
        m_key = tuple(par)
        m_key_str = "/".join(m_key)
        if m_key not in model_flat:
            logging.warning("%s in checkpoint doesn't exist in model. Skip.", m_key_str)
            print(m_key_str)
            adhoc_match(m_key_str, m_params, model_flat)
            continue
        if skip_regex and re.findall(skip_regex, m_key_str):
            logging.info("Skip loading parameter %s.", m_key_str)
            continue
        logging.info("Loading %s from checkpoint into model", m_key_str)
        model_flat[m_key] = m_params

    return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))



class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: Tuple[int, int] = (1, 1)
  dtype: jnp.dtype = jnp.float32
  bottleneck: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True, add_relu: bool = True) -> jnp.ndarray:
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    nout = self.filters * 4 if self.bottleneck else self.filters

    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = conv(nout, (1, 1), self.strides, name='proj_conv')(residual)
      residual = batch_norm(name='proj_bn')(residual)

    if self.bottleneck:
      x = conv(self.filters, (1, 1), name='conv1')(x)
      x = batch_norm(name='bn1')(x)
      x = nn_layers.IdentityLayer(name='relu1')(nn.relu(x))

    y = conv(
        self.filters, (3, 3),
        self.strides,
        padding=[(1, 1), (1, 1)],
        name='conv2')(
            x)
    y = batch_norm(name='bn2')(y)
    y = nn_layers.IdentityLayer(name='relu2')(nn.relu(y))

    if self.bottleneck:
      y = conv(nout, (1, 1), name='conv3')(y)
    else:
      y = conv(nout, (3, 3), padding=[(1, 1), (1, 1)], name='conv3')(y)
    y = batch_norm(name='bn3', scale_init=nn.initializers.zeros)(y)
    if add_relu:
        y = nn_layers.IdentityLayer(name='relu3')(nn.relu(residual + y))
    
    return y, residual

BLOCK_SIZE_OPTIONS = {
    5: ([1], True),  # Only strided blocks. Total stride 4.
    8: ([1, 1], True),  # Only strided blocks. Total stride 8.
    11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
    14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
    9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True),
} 

ADAPTERS = {
   "bottleneck": adapters.BottleneckAdapterParallel,
    "MHSA":  adapters.MHSAAdapterParallel,
    "channel_MHSA":  adapters.MHSAChannelsAdapterParallel,
    "cross_attention":  adapters.CrossAttentionAdapter
}

def get_adapter(adapter_name, adapter_config): # --> Type[nn.Module]:
    if adapter_name is None:
        return None
    adapter = ADAPTERS[adapter_name]
    return adapter(**adapter_config[adapter_name])


class ResNet(nn.Module):
    """ResNet architecture.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    num_filters: Num filters.
    num_layers: Num layers.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """

    adapter_name: str
    adapter_config: Any
    num_outputs: Optional[int]
    num_filters: int = 64
    num_layers: int = 50
    kernel_init: Callable[..., Any] = initializers.lecun_normal()
    bias_init: Callable[..., Any] = initializers.zeros
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = False, debug: bool = False
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
       Un-normalized logits.
    """
        if self.num_layers not in BLOCK_SIZE_OPTIONS:
            raise ValueError("Please provide a valid number of layers")
        block_sizes, bottleneck = BLOCK_SIZE_OPTIONS[self.num_layers]
         # <MOD>

        batch_norm = functools.partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        x = nn.Conv(
            self.num_filters,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            use_bias=False,
            dtype=self.dtype,
            name="stem_conv",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            name="init_bn",
        )(x)
        x = nn_layers.IdentityLayer(name="init_relu")(nn.relu(x))
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

        residual_block = functools.partial(
            ResidualBlock, dtype=self.dtype, bottleneck=bottleneck
        )
        representations = {"stem": x}
        
        
        adapter = get_adapter(self.adapter_name, self.adapter_config)

        for i, block_size in enumerate(block_sizes):
            strides_adapter = (2, 2) if i > 0 else (1, 1)

            x_copy = x.copy()
            print(x.shape[-1] if i == 0 else x.shape[-1] * 2)
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                filters = self.num_filters * 2 ** i

                if adapter is None:
                    add_relu = True
                else:
                    #if adapter, wedon't perform reLU before adding the residual and the output of the adapter
                    add_relu=j != block_size - 1
                x, residual = residual_block(filters=filters, strides=strides)(
                    x, train, add_relu)
                
            
            representations[f"stage_{i + 1}"] = x
            representations[f"stage_residual_{i + 1}"] = x
            if  adapter is not None:
                if self.adapter_config.shared:
                    y = adapter(x_copy, deterministic=not train)
                    representations[f"stage_adapter_{i + 1}"] = y
                    x = nn_layers.IdentityLayer(name="relu3" + f"_{i}")(
                        nn.relu(y + x + residual)
                    )
                else:
                    y = get_adapter(self.adapter_name, self.adapter_config)(x_copy, deterministic=not train)
                    representations[f"stage_adapter_{i + 1}"] = y
                    x = nn_layers.IdentityLayer(name="relu3" + f"_{i}")(
                        nn.relu(y + x + residual)
                    )

        if self.num_outputs:
            x = jnp.mean(x, axis=(1, 2))
            x = nn_layers.IdentityLayer(name="pre_logits")(x)
            x = nn.Dense(
                self.num_outputs,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                name="output_projection",
            )(x)
            return x
        else:
            return representations
        
#adapter_layers_ids = ids_str2ints(self.adapter_layers) 

def init_from_model_state(
    train_state: Any,
    pretrain_state: Any,
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None,
):
    """Updates the train_state with data from pretrain_state.

    Args:
        train_state: A raw TrainState for the model.
        pretrain_state: A TrainState that is loaded with parameters/state of
          a  pretrained model.
        ckpt_prefix_path: Prefix to restored model parameters.
        model_prefix_path: Prefix to the parameters to replace in the subtree model.
        name_mapping: Mapping from parameter names of checkpoint to this model.
        skip_regex: If there is a parameter whose parent keys match the regex,
          the parameter will not be replaced from pretrain_state.

    Returns:
        Updated train_state.
    """
    name_mapping = name_mapping or {}
    restored_params = pretrain_state["params"]
    restored_model_state = pretrain_state["model_state"]
    # TODO(scenic): Add support for optionally restoring optimizer state.
    if (
        restored_model_state is not None
        and train_state.model_state is not None
        and train_state.model_state
    ):
        if model_prefix_path:
            # Insert model prefix after 'batch_stats'.
            model_prefix_path = ["batch_stats"] + model_prefix_path
            if "batch_stats" in restored_model_state:
                ckpt_prefix_path = ckpt_prefix_path or []
                ckpt_prefix_path = ["batch_stats"] + ckpt_prefix_path
        elif "batch_stats" not in restored_model_state:  # Backward compatibility.
            model_prefix_path = ["batch_stats"]
        if ckpt_prefix_path and ckpt_prefix_path[0] != "batch_stats":
            ckpt_prefix_path = ["batch_stats"] + ckpt_prefix_path

        model_state = _replace_dict(
            train_state.model_state,
            restored_model_state,
            ckpt_prefix_path,
            model_prefix_path,
            name_mapping,
            skip_regex,
        )
    # train_state = train_state.replace(  # pytype: disable=attribute-error
    #    model_state=model_state)
    return model_state

def _get_default_configs_for_testing() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(num_filters=16, num_layers=5, data_dtype_str="float32",)
    )

class ResNetClassificationAdapterModel(ClassificationModel):
    """Implements the ResNet model for classification."""

    def build_flax_model(self) -> nn.Module:
        model_dtype = getattr(jnp, self.config.get("model_dtype_str", "float32"))
        return ResNet(
            adapter_name=self.config.adapter_name,
            adapter_config=self.config.adapter_config,
            num_outputs=self.dataset_meta_data["num_classes"],
            num_filters=self.config.num_filters,
            num_layers=self.config.num_layers,
            dtype=model_dtype,
        )
    
    def default_flax_model_config(self) -> ml_collections.ConfigDict:
        return _get_default_configs_for_testing()
    
    def init_from_train_state(
        self,
        train_state: Any,
        restored_train_state: Any,
        model_conf: ml_collections.ConfigDict,
        ) -> Any:
        """Updates the train_state with data from `restored_train_state`.

            This function is writen to be used for 'fine-tuning' experiments. Here, we
            do some surgery to support larger resolutions (longer sequence length) in
            the transformer block, with respect to the learned pos-embeddings.

            Args:
            train_state: A raw TrainState for the model.
            restored_train_state: A TrainState that is loaded with parameters/state of
                a pretrained model.
            restored_model_cfg: Configuration of the model from which the
                `restored_train_state` come from. Usually used for some asserts.

            Returns:
            Updated train_state.
        """
       
        print("Initializing model from pretrained model")

        with open('/network/scratch/t/tengmeli/scratch/debug/weights_from_GLC.pkl', 'rb') as handle:
            restored = pickle.load(handle)
        match = {'ResidualBlock_0': 'layer1.0',
         'ResidualBlock_1': 'layer1.1',
         'ResidualBlock_2': 'layer1.2',
         'ResidualBlock_3': 'layer2.0',
         'ResidualBlock_4': 'layer2.1',
         'ResidualBlock_5': 'layer2.2',
         'ResidualBlock_6': 'layer2.3',
         'ResidualBlock_7': 'layer3.0',
         'ResidualBlock_8': 'layer3.1',
         'ResidualBlock_9': 'layer3.2',
         'ResidualBlock_10': 'layer3.3',
         'ResidualBlock_11': 'layer3.4',
         'ResidualBlock_12': 'layer3.5',
         'ResidualBlock_13': 'layer4.0',
         'ResidualBlock_14': 'layer4.1',
         'ResidualBlock_15': 'layer4.2'}
        #import pdb; pdb.set_trace()
        if hasattr(train_state, "optimizer"):
            # TODO(dehghani): Remove support for flax optim.
            params = flax.core.unfreeze(train_state.optimizer.target)
        else:
            params = flax.core.unfreeze(train_state.params)
        for pname, pvalue in params.items():
    
            if pname == "output_projection":
                # params[pname] = {"kernel": jnp.transpose(jnp.array(restored["fc.weight"])), "bias": jnp.array(restored["fc.bias"])}
                continue
            else:

                if pname == "stem_conv":
                    aa = params[pname]["kernel"].copy()
                    aa = aa.at[:, :, :3, :].set(jnp.transpose(jnp.array(restored["conv1.weight"])))
                    if model_conf.init_new_channel_zero:
                        print("init new channels to zero")
                        aa = aa.at[:, :, 3:, :].set(0)

                else:
                    if pname.startswith("ResidualBlock"):
                        layername = match[pname]
                        subkeys = pvalue.keys()
                        for key in subkeys:
                            if key.startswith("conv"):
                                params[pname][key] = {"kernel": jnp.transpose(jnp.array(restored[layername+"."+key+".weight"]), (2,3,1,0))}
                            elif key.startswith("proj_conv"):
                                nn =layername +".downsample.0.weight"
                                params[pname][key] = {"kernel": jnp.transpose(jnp.array(restored[nn]), (2,3,1,0))}
                            elif  key.startswith("proj_bn"):
                                  params[pname][key] = {"scale":jnp.array(restored[layername +".downsample.1.weight"]),
                                                     "bias":jnp.array(restored[layername +".downsample.1.bias"])}
                            else: #bn
                                params[pname][key] = {"scale": jnp.transpose(jnp.array(restored[layername+"."+key+".weight"])),
                                                      "bias":jnp.array(restored[layername+"."+key+".bias"])}
                    else:
                        print("skip param", pname)

        logging.info("Parameter summary after initialising from train state:")
        debug_utils.log_param_shapes(params)
        if hasattr(train_state, "optimizer"):
            # TODO(dehghani): Remove support for flax optim.
            return train_state.replace(
                optimizer=train_state.optimizer.replace(
                    target=flax.core.freeze(params)
                ),
                model_state=restored_train_state.model_state,
            )
        else:
            model_state = init_from_model_state(train_state, restored_train_state)
            return train_state.replace(
                params=flax.core.freeze(params), model_state=model_state
            )
