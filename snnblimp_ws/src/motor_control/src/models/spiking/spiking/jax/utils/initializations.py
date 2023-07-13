import jax
import jax.numpy as jnp

from spiking.jax.utils.typing import Dtype, InitFn


def constant(scale: float, dtype: Dtype = jnp.float_) -> InitFn:
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return scale * jnp.ones(shape, dtype)

    return init
