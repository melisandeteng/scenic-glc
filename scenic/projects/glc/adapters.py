import flax
import flax.linen as nn
import jax
from scenic.model_lib.layers import nn_layers
from typing import Tuple, Callable, Any, Optional, Union, Dict
import ml_collections
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
import jax.numpy as jnp
import functools
from jax.nn import initializers
import einops
from typing import Any, Callable, Optional, Tuple
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import DenseGeneral
from flax.linen.linear import PrecisionLike
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.linen.module import merge_param

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class BottleneckAdapterSeries(nn.Module):
    adapter_dim: int

    @nn.compact
    def __call__(self, x):
        hidden_dim = x.shape[-1]
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.adapter_dim)(y)
        y = nn.gelu(y)
        # Default initalization.
        # y = nn.Dense(hidden_dim)(y)
        # Initialization from https://arxiv.org/pdf/1902.00751.pdf
        # y = nn.Dense(hidden_dim, kernel_init=nn.initializers.normal(stddev=1e-3))(y)
        # Zero Initialization so that added adapter does not change the representation.
        y = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.zeros)(y)
        return x + y  # Residual.


class BottleneckAdapterParallel(nn.Module):
    hidden_dim: int
    adapter_dim: int
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        # import pdb; pdb.set_trace()
        hidden_dim = self.hidden_dim
        adapter_dim = self.adapter_dim
        strides = self.strides
        if hidden_dim is None:
            hidden_dim = x.shape[-1]
        # y = nn.LayerNorm()(x)
        y = nn.Conv(
            adapter_dim,
            kernel_size=(1, 1),
            strides=strides,
            use_bias=True,
            dtype=x.dtype,
            name="dense_1",
        )(x)
        # nn.Dense(self.adapter_dim, name="dense_1_adapter")(x)
        y = nn_layers.IdentityLayer(name="gelu")(nn.gelu(y))
        # Default initalization.
        # y = nn.Dense(hidden_dim)(y)
        # Initialization from https://arxiv.org/pdf/1902.00751.pdf
        # y = nn.Dense(hidden_dim, kernel_init=nn.initializers.normal(stddev=1e-3))(y)
        # Zero Initialization so that added adapter does not change the representation.
        y = nn.Dense(
            hidden_dim, kernel_init=jax.nn.initializers.zeros, name="dense_2_adapter"
        )(y)
        return y  # Residual.


class MHSAAdapterParallel(nn.Module):
    """Transformer encoder layer.

  Attributes:

    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value.

  Returns:
  output after transformer encoder block.
  """

    # mlp_dim: int
    num_heads: int
    # filters: int
    # strides: Any
    dtype: Any = jnp.float32
    qkv_features: int = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    stochastic_depth: float = 0.0
    bottleneck: bool = True
    runavg: bool = True
    out_features: Any = None
    shared_MHSA: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Applies Encoder1DBlock module.

        Args:
          inputs: Input data.
          deterministic: Deterministic or not (to apply dropout).

        Returns:
          Output after transformer encoder block.
        """
        # Attention block.
        # assert inputs.ndim == 3
        # needs_projection = inputs.shape[-1] != self.filters * 4 or self.strides != (1, 1)

        batch_norm = functools.partial(
            nn.BatchNorm,
            use_running_average=self.runavg,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

        x = inputs
        print("inputs to MHSA shape", x.shape)
        shape_in = inputs.shape
        if len(x.shape) == 4:
            x = einops.rearrange(inputs, "n h w c -> n (h w) c")
        # if needs_projection:
        #  print("needs projection")
        #  residual = conv(nout, (1, 1), self.strides, name='proj_conv')(residual)
        #  residual = batch_norm(name='proj_bn')(residual)
        if self.shared_MHSA is False:
            x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            out_features=self.out_features,
        )(x, x)
        x = einops.rearrange(x, "n (h w) c -> n h w c", h=shape_in[1], w=shape_in[2])
        print("outputs to MHSA shape", x.shape)
        # x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
        # x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
        # if self.out_features != inputs.shape[-1]:
        #    print("Projecting adapter output to match residual size")
        #    x = conv(residual.shape[-1], (1, 1), name='proj_conv')(x)
        #    print("outputs projection in MHSA shape", x.shape)
        # remove x = x+residual because this is the adapter block
        # x = x + residual
        # nout = self.filters * 4 if self.bottleneck else self.filters
        # x = conv(nout, (1, 1), self.strides, name='proj_conv')(x)

        return x


class MHSAChannelsAdapterParallel(nn.Module):
    """self attention adapter across channels

    Attributes:
      num_heads: Number of self-attention heads.
      dtype: The dtype of the computation (default: float32).
      qkv_features: Number of features for query, key, and value.
      attention_dropout_rate: Dropout for attention heads.
      deterministic: Whether to use deterministic or stochastic
      out_features: Number of features for output.

    Returns:
    output after transformer encoder block.
    """

    num_heads: int
    dtype: Any = jnp.float32
    qkv_features: int = None
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    out_features: Any = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        # flatten h,w

        shape_in = inputs.shape

        x = einops.rearrange(inputs, "n h w c -> n (h w) c")
        x = nn.LayerNorm(dtype=self.dtype)(x)
        # transpose
        x = jnp.swapaxes(x, 1, 2)  # n c (h w)
        # perform attention channel wise

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            out_features=self.out_features,
        )(x, x)

        # transpose back
        x = jnp.swapaxes(x, 1, 2)  # (B, T, C, F)

        x = einops.rearrange(x, "n (h w) c -> n h w c", h=shape_in[1], w=shape_in[2])
        return x


def zeros_init() -> jax.nn.initializers.Initializer:
    """Builds an initializer that returns a constant array full of zeros.

  >>> import jax, jax.numpy as jnp
  >>> from flax.linen.initializers import zeros_init
  >>> zeros_initializer = zeros_init()
  >>> zeros_initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)
  Array([[0., 0., 0.],
         [0., 0., 0.]], dtype=float32)
  """
    return jax.nn.initializers.zeros


class CustomMultiHeadAttention(nn.Module):
    """_summary_
    Perform cross attention where the query comes from one input and the key and values are initialized from scratch.
    Also kv_length and q_length may be different.
    we will perform Attention(Q input, K, V) where K and V are initialized from scratch.
    Args:
        nn (_type_): _description_
    """

    num_heads: int
    head_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = nn.dot_product_attention
    kv_features: Optional[int] = None  # number of features for key and value
    kv_initialization: Any = nn.initializers.xavier_uniform()
    broadcast_dropout: bool = False
    out_feats: Optional[int] = 64

    @nn.compact
    def __call__(
        self, inputs_q: Array, deterministic: Optional[bool] = None,
    ):
        features_in = inputs_q.shape[-1]

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, self.head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            # dot_general=jax.lax.dot_general,
        )
        query = dense(name="query")(inputs_q)
        """
      def kernel_init_wrap(rng, shape, dtype=jnp.float32):
        flat_shape = (np.prod(shape[:n_batch_dims]) *
                      np.prod(shape[n_batch_dims:n_axis + n_batch_dims]),
                      np.prod(shape[-n_features:]),)
        flat_shape = jax.tree_map(int, flat_shape)
        kernel = self.kernel_init(rng, flat_shape, dtype)
        if isinstance(kernel, meta.AxisMetadata):
          return meta.replace_boxed(kernel, jnp.reshape(kernel.unbox(), shape))
        return jnp.reshape(kernel, shape)
      """

        shape_kv = tuple([self.kv_features, self.num_heads, self.head_dim])
        print("SHAPE KV", shape_kv)
        """ 
        key = self.param(
            name="key",
            init_fn=self.kv_initialization,
            shape=shape_kv,
            param_dtype=self.param_dtype,
        )
        value = self.param(
           name="value",
            init_fn=self.kv_initialization,
            shape=shape_kv,
            param_dtype=self.param_dtype,
        )
        """
        # key: keys for calculating attention with shape of
        # `[batch..., kv_length, num_heads, qk_depth_per_head]`.

        key = self.param("key", self.kv_initialization, shape_kv, self.param_dtype)
        key = jnp.tile(key, (query.shape[0], 1, 1, 1))

        # value: values to be used in attention with shape of
        # `[batch..., kv_length, num_heads, v_depth_per_head]`.
        value = self.param(
            "value",
            self.kv_initialization,
            (
                self.kv_features,
                self.num_heads,
                self.head_dim
                # self.num_heads, int(self.out_feats / self.num_heads)]
            ),
            self.param_dtype,
        )

        value = jnp.tile(
            key, (query.shape[0], 1, 1, 1)
        )  # jnp.stack([value] * query.shape[0])
        dropout_rng = None
        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param(
                "deterministic", self.deterministic, deterministic
            )
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True
        # cross attention

        x = nn.dot_product_attention(
            query,
            key,
            value,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )

        out = DenseGeneral(
            features=self.out_feats,  # features_in,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            # dot_general=jax.lax.dot_general,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out


class CrossAttentionAdapter(nn.Module):
    """
    _summary_
    Adapter inspired from Perceiver IO 
    in this type of adapter, we perform cross attention between the output of the previous layer (query) 
    and learned free embeddings learned from scratch (keys, values) initialized ~ 0 
    Args:
      
    """

    num_heads: int
    head_dim: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    dropout_rate: float = 0.0
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = nn.dot_product_attention
    kv_features: Optional[int] = None  # number of features for key and value
    kv_initialization: Any = nn.initializers.xavier_uniform()
    out_feats: Optional[int] = 64

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        shape_in = inputs.shape

        x = einops.rearrange(inputs, "n h w c -> n (h w) c")
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = CustomMultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            dropout_rate=self.dropout_rate,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            attention_fn=self.attention_fn,
            kv_features=self.kv_features,
            kv_initialization=self.kv_initialization,
            out_feats=self.out_feats,
        )(x, deterministic)
        # perhaps need to rearrange dimensions
        x = einops.rearrange(x, "n (h w) c -> n h w c", h=shape_in[1], w=shape_in[2])
        return x


"""
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        u = inputs.shape[1]
        kv_inputs = jax.lax.expand_dims(jnp.identity(u, dtype=self.dtype), [0, -1])
        kv_inputs = jnp.repeat(kv_inputs, inputs.shape[0], axis=0)
        kv_inputs = jnp.repeat(kv_inputs, inputs.shape[-1], axis=-1)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),  # should I have a different kernel init for q and kv?
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            out_features=self.out_features,
        )(x, kv_inputs)
        return x

 """
