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

    hidden_dim = self.hidden_dim
    adapter_dim = self.adapter_dim
    strides = self.strides
    if hidden_dim is None: 
        hidden_dim = x.shape[-1]
    #y = nn.LayerNorm()(x)
    y = nn.Conv(
        adapter_dim,
        kernel_size=(1, 1),
        strides=strides,
        use_bias=True,
        dtype=x.dtype,
        name='dense_1')(
            x)
    #nn.Dense(self.adapter_dim, name="dense_1_adapter")(x)
    y = nn_layers.IdentityLayer(name='gelu')(nn.gelu(y))
    # Default initalization.
    # y = nn.Dense(hidden_dim)(y)
    # Initialization from https://arxiv.org/pdf/1902.00751.pdf
    # y = nn.Dense(hidden_dim, kernel_init=nn.initializers.normal(stddev=1e-3))(y)
    # Zero Initialization so that added adapter does not change the representation.
    y = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.zeros, name="dense_2_adapter")(y)
    return y  # Residual.




class MHSAAdapterParallel(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value.

  Returns:
  output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  filters: int
  strides: Any
  dtype: Any = jnp.float32
  qkv_features: int = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  bottleneck: bool=True
  runavg: bool=True

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
    #assert inputs.ndim == 3
    needs_projection = inputs.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    nout = self.filters * 4 if self.bottleneck else self.filters
    
    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=self.runavg,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    residual = inputs
    #if needs_projection:
    #  print("needs projection")
    #  residual = conv(nout, (1, 1), self.strides, name='proj_conv')(residual)
    #  residual = batch_norm(name='proj_bn')(residual)

      #residual = batch_norm(name='proj_bn')(residual)
    x = nn.LayerNorm(dtype=self.dtype)(residual)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features = self.qkv_features,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + residual
    x = conv(nout, (1, 1), self.strides, name='proj_conv')(x)

    return(x)
    """
    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    
    return y + x
    """
