"""Implementation of ResNet and adapters."""

import functools
from typing import Tuple, Callable, Any,  List,Optional, Union, Dict, Mapping

import scenic.train_lib.train_utils as train_utils

from absl import logging
import flax
import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import nn_layers

from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils

from scenic.projects.glc.adapters import  BottleneckAdapterParallel
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
import comet_ml
PyTree = Union[Mapping[str, Mapping], Any]
def ids_str2ints(ids_str):
  return [int(v) for v in ids_str.split('_')] if ids_str else []
def ids_ints2str(ids_ints):
  return '_'.join([str(v) for v in sorted(ids_ints)])

def adhoc_match(m_key_str, m_params, model_flat):
    print("match adhoc")
    if m_key_str == "batch_stats/ResidualBlock_2/bn3/var":
         model_flat[tuple(['batch_stats','bn3block_0', 'var'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_2/bn3/mean":
        model_flat[tuple(['batch_stats','bn3block_0', 'mean'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_6/bn3/var":
         model_flat[tuple(['batch_stats','bn3block_1', 'var'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_6/bn3/mean":
        model_flat[tuple(['batch_stats','bn3block_1', 'mean'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_12/bn3/var":
         model_flat[tuple(['batch_stats','bn3block_2', 'var'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_12/bn3/mean":
        model_flat[tuple(['batch_stats','bn3block_2', 'mean'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_15/bn3/var":
         model_flat[tuple(['batch_stats','bn3block_3', 'var'])] = m_params
    elif m_key_str == "batch_stats/ResidualBlock_15/bn3/mean":
        model_flat[tuple(['batch_stats','bn3block_3', 'mean'])] = m_params
    return(model_flat)

def _replace_dict(model: PyTree,
                  restored: PyTree,
                  ckpt_prefix_path: Optional[List[str]] = None,
                  model_prefix_path: Optional[List[str]] = None,
                  name_mapping: Optional[Mapping[str, str]] = None,
                  skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with restored ones from checkpoint."""
  #import pdb; pdb.set_trace()
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
      dict(restored), keep_empty_nodes=True)
  model_flat = flax.traverse_util.flatten_dict(
      dict(model), keep_empty_nodes=True)
  #import pdb;pdb.set_trace()
# ('/init_bn').strip("/").split("/")
  for m_key, m_params in restored_flat.items():
    
    # pytype: disable=attribute-error
    for name, to_replace in name_mapping.items():
      m_key = tuple(to_replace if k == name else k for k in m_key)
    # pytype: enable=attribute-error
    par = []
    for elem in m_key:
        elem = elem.strip('/').split('/')
        par += elem
    m_key = tuple(par)
    m_key_str = '/'.join(m_key)
    if m_key not in model_flat:
      logging.warning('%s in checkpoint doesn\'t exist in model. Skip.',
                      m_key_str)
      print(m_key_str)
      adhoc_match(m_key_str, m_params, model_flat)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))



class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: Tuple[int, int] = (1, 1)
  dtype: jnp.dtype = jnp.float32
  bottleneck: bool = True
  
  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True, add_bn=True) -> jnp.ndarray:
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
    if add_bn: 
        y = batch_norm(name='bn3', scale_init=nn.initializers.zeros)(y)
        y = nn_layers.IdentityLayer(name='relu3')(nn.relu(residual + y))
    return y, residual



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
  adapter_layers: str
  adapter_dim : int
  num_outputs: Optional[int]
  num_filters: int = 64
  num_layers: int = 50
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32
  
    

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False,
      debug: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
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
      raise ValueError('Please provide a valid number of layers')
    block_sizes, bottleneck = BLOCK_SIZE_OPTIONS[self.num_layers]
    adapter_layers_ids = ids_str2ints(self.adapter_layers)  # <MOD>
    
    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)

    x = nn.Conv(
        self.num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='stem_conv')(
            x)
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='init_bn')(
            x)
    x = nn_layers.IdentityLayer(name='init_relu')(nn.relu(x))
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
   
    residual_block = functools.partial(
        ResidualBlock, dtype=self.dtype, bottleneck=bottleneck)
    representations = {'stem': x}
    for i, block_size in enumerate(block_sizes):
      strides_adapter = (2, 2) if i > 0 else (1,1)
      
     # y =  BottleneckAdapterParallel(
     #       adapter_dim=self.adapter_dim,
     #       hidden_dim = x.shape[-1] if i==0 else x.shape[-1]*2, 
     #       strides = strides_adapter, 
     #       )(x)  # MOD>
      x_copy = x.copy()
      print(x.shape[-1] if i==0 else x.shape[-1]*2)
        #name=f'residual_adapter_{i}',
      for j in range(block_size):
          strides = (2, 2) if i > 0 and j == 0 else (1, 1)
          filters = self.num_filters * 2**i
    
          x, residual = residual_block(filters=filters, strides=strides)(x, train,add_bn=(j!=block_size-1))
         # if j != (block_size-1):
         #     y = batch_norm(name='bn3', scale_init=nn.initializers.zeros)(x)
          #    x = nn_layers.IdentityLayer(name='relu3')(nn.relu(residual + y))
      representations[f'stage_{i + 1}'] = x
      #print(x.shape)
      
      y =  BottleneckAdapterParallel(
            adapter_dim=self.adapter_dim,
            hidden_dim = x.shape[-1], # if i==0 else x.shape[-1]*2, 
            strides = strides_adapter, 
            )(x_copy)  
      #print(y.shape)
      #print("hidden_dim", x.shape[-1] if i==0 else x.shape[-1]*2)
      x = x+y
      x = batch_norm(name='bn3'+ f"block_{i}", scale_init=nn.initializers.zeros)(x)
      x = nn_layers.IdentityLayer(name='relu3'+ f"_{i}")(nn.relu(residual + x))

    # Head.
    if self.num_outputs:
      x = jnp.mean(x, axis=(1, 2))
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='output_projection')(
              x)
      return x
    else:
      return representations


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model. The second argument indicates whether we
# use ba5 layers or not.
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
    200: ([3, 24, 36, 3], True)
}

def init_from_model_state(
    train_state:Any,
    pretrain_state: Any,
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None) :
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
    restored_params = pretrain_state['params']
    restored_model_state = pretrain_state['model_state']
    # TODO(scenic): Add support for optionally restoring optimizer state.
    if (restored_model_state is not None and train_state.model_state is not None and train_state.model_state):
      if model_prefix_path:
      # Insert model prefix after 'batch_stats'.
        model_prefix_path = ['batch_stats'] + model_prefix_path
        if 'batch_stats' in restored_model_state:
            ckpt_prefix_path = ckpt_prefix_path or []
            ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
      elif 'batch_stats' not in restored_model_state:  # Backward compatibility.
        model_prefix_path = ['batch_stats']
      if ckpt_prefix_path and ckpt_prefix_path[0] != 'batch_stats':
        ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path

      model_state = _replace_dict(train_state.model_state,
                                restored_model_state,
                                ckpt_prefix_path,
                                model_prefix_path,
                                name_mapping,
                                skip_regex)
    #train_state = train_state.replace(  # pytype: disable=attribute-error
    #    model_state=model_state)
    return model_state



class ResNetClassificationAdapterModel(ClassificationModel):
  """Implemets the ResNet model for classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ResNet(
          adapter_layers = self.config.adapter_layers,
          adapter_dim= self.config.adapter_dim ,
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()



  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
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
    print("INITIALIZING RESNET ADAPTER MODEL")
    del restored_model_cfg
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      params = flax.core.unfreeze(train_state.optimizer.target)
      restored_params = flax.core.unfreeze(
          restored_train_state.optimizer.target)
    else:
      params = flax.core.unfreeze(train_state.params)
      restored_params = flax.core.unfreeze(restored_train_state["params"])
    #import pdb; pdb.set_trace()
    for pname, pvalue in restored_params.items():
      
      if pname == 'output_projection':
        # The `output_projection` is used as the name of the linear layer at the
        # head of the model that maps the representation to the label space.
        # By default, for finetuning to another dataset, we drop this layer as
        # the label space is different.
        continue
      else:

        if pname=="stem_conv":
            print("RESTORING STEM CONV RGB CHANNELS FROM INIT")

            aa =params[pname]["kernel"].copy()
            aa = aa.at[:,:,:3,:].set(pvalue["kernel"])
            params[pname] = {"kernel":aa}

        else:
            if pname in params.keys():
                if params[pname].keys() == pvalue.keys():
                    print("loading param ", pname, " into model")
            
                    params[pname] = pvalue
                else:
                    if "bn3" in pvalue.keys() and "bn3" not in params[pname].keys():
                        bn3 = pvalue.pop("bn3")
                        if pname == "ResidualBlock_2":
                            params["bn3block_0"] = bn3
                        elif pname == "ResidualBlock_6":
                            params["bn3block_1"] = bn3
                        elif pname == "ResidualBlock_12":
                            params["bn3block_2"] = bn3
                        elif pname == "ResidualBlock_15":
                            params["bn3block_3"] = bn3
                        else:
                            print("ERROR")
                    params[pname] = pvalue
            else:
                print("skip param", pname)


    logging.info('Parameter summary after initialising from train state:')
    debug_utils.log_param_shapes(params)
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      return train_state.replace(
          optimizer=train_state.optimizer.replace(
              target=flax.core.freeze(params)),
          model_state=restored_train_state.model_state)
    else:
      model_state = init_from_model_state(train_state,restored_train_state)
      return train_state.replace(
          params=flax.core.freeze(params) ,
          model_state=model_state)


class ResNetClassificationModel(ClassificationModel):
  """Implemets the ResNet model for classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ResNet(
      
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()


  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
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
    del restored_model_cfg
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      params = flax.core.unfreeze(train_state.optimizer.target)
      restored_params = flax.core.unfreeze(
          restored_train_state.optimizer.target)
    else:
      params = flax.core.unfreeze(train_state.params)
      restored_params = flax.core.unfreeze(restored_train_state.params)
    for pname, pvalue in restored_params.items():
      if pname == 'output_projection':
        # The `output_projection` is used as the name of the linear layer at the
        # head of the model that maps the representation to the label space.
        # By default, for finetuning to another dataset, we drop this layer as
        # the label space is different.
        continue
      else:
        params[pname] = pvalue
    logging.info('Parameter summary after initialising from train state:')
    debug_utils.log_param_shapes(params)
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      return train_state.replace(
          optimizer=train_state.optimizer.replace(
              target=flax.core.freeze(params)),
          model_state=restored_train_state.model_state)
    else:
      return train_state.replace(
          params=flax.core.freeze(params),
          model_state=restored_train_state.model_state)


class ResNetMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Implemets the ResNet model for multi-label classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()


def _get_default_configs_for_testing() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          num_filters=16,
          num_layers=5,
          data_dtype_str='float32',
      ))




"""Training Script."""


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
    comet_exp: None,
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str, Any]]:
    """Runs a single step of training.

    Given the state of the training and a batch of data, computes
    the loss and updates the parameters of the model.

    Note that in this code, the buffers of the first (train_state) and second
    (batch) arguments are donated to the computation.

    Args:
      train_state: The state of training including the current global_step,
        model_state, rng, params, and optimizer. The buffer of this argument can
        be donated to the computation.
      batch: A single batch of data. The buffer of this argument can be donated to
        the computation.
      flax_model: A Flax model.
      loss_fn: A loss function that given logits, a batch, and parameters of the
        model calculates the loss.
      lr_fn: The learning rate fn used for the logging the learning rate.
      metrics_fn: A metrics function that given logits and batch of data,
        calculates the metrics as well as the loss.
      config: Configurations of the experiment.
      debug: Whether the debug mode is enabled during training. `debug=True`
        enables model specific logging/storing some values using
        jax.host_callback.

    Returns:
      Updated state of training and computed metrics and some training logs.
    """
    #  import pdb; pdb.set_trace()
    training_logs = {}
    #print("=======================================================")
    #print("batch inputs")
    #print(batch["inputs"].shape)
    #print("========================================================")
    new_rng, rng = jax.random.split(train_state.rng)

    if config.get("mixup") and config.mixup.alpha:
        mixup_rng, rng = jax.random.split(rng, 2)
        mixup_rng = train_utils.bind_rng_to_host_device(
            mixup_rng, axis_name="batch", bind_to=config.mixup.get("bind_to", "device")
        )
        batch = dataset_utils.mixup(
            batch,
            config.mixup.alpha,
            config.mixup.get("image_format", "NHWC"),
            rng=mixup_rng,
        )

    # Bind the rng to the host/device we are on.
    dropout_rng = train_utils.bind_rng_to_host_device(
        rng, axis_name="batch", bind_to="device"
    )

    def training_loss_fn(params, fixed_params):
        variables = {"params": params, **train_state.model_state}
        logits, new_model_state = flax_model.apply(
            variables,
            batch["inputs"],
            mutable=["batch_stats"],
            train=True,
            rngs={"dropout": dropout_rng},
            debug=debug,
        )
        loss = loss_fn(logits, batch, variables["params"])
        return loss, (new_model_state, logits)

    compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
    (train_cost, (new_model_state, logits)), grad = compute_gradient_fn(
        train_state.params
    )
    #  if comet_exp is not None:
    #    comet_exp.log_metric("train_loss", train_cost, step=train_state.global_step, include_context=True)
    del train_cost
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = jax.lax.pmean(grad, axis_name="batch")

    if config.get("max_grad_norm") is not None:
        grad = clip_grads(grad, config.max_grad_norm)

    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)

    training_logs["l2_grads"] = jnp.sqrt(
        sum([jnp.vdot(g, g) for g in jax.tree_leaves(grad)])
    )
    ps = jax.tree_leaves(new_params)
    training_logs["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
    us = jax.tree_leaves(updates)
    training_logs["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
    # TODO(dehghani): Can we get this from the optimizer instead?
    training_logs["learning_rate"] = lr_fn(train_state.global_step)

    metrics = metrics_fn(logits, batch)

    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        global_step=train_state.global_step + 1,
        opt_state=new_opt_state,
        params=new_params,
        model_state=new_model_state,
        rng=new_rng,
    )

    return new_train_state, metrics, training_logs

