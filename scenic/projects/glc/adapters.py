import flax
import flax.linen as nn
import jax 
from scenic.model_lib.layers import nn_layers
from typing import Tuple, Callable, Any, Optional, Union, Dict
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


