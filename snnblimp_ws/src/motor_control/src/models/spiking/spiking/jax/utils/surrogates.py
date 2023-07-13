from typing import Callable, Tuple, Type

import flax.linen as nn
import jax
from jax import custom_gradient, custom_jvp, lax
import jax.numpy as jnp

from spiking.jax.utils.typing import Array


@custom_gradient
def spike_atan_old(x: Array) -> Tuple[Array, Callable[[Array], Array]]:
    """
    Example of 'old' implementation of custom gradients (incompatible with forward-mode AD).
    """
    dtype = jax.dtypes.canonicalize_dtype(jnp.float_)
    s = lax.gt(x, 0.0).astype(dtype)

    def sg(grad):
        return (grad * (1 / (1 + 10 * x * x)),)

    return s, sg


# custom_jvp is compatible with forward-mode AD, custom_vjp/custom_gradient is not
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
@custom_jvp
def spike_atan(x: Array) -> Array:
    """
    Spike function (Heaviside) with derivative of arctan surrogate gradient.

    From "Incorporating Learnable Membrane Time Constant...", Fang et al., arXiv 2020.

    TODO:
    - parameterize
    - JIT
    - benchmark thresholding with GPU
    - impact of dtype selection
    """
    dtype = jax.dtypes.canonicalize_dtype(jnp.float_)
    s = lax.gt(x, 0.0).astype(dtype)
    return s


@spike_atan.defjvp
def spike_atan_jvp(primals, tangents):
    (x,) = primals
    (grad,) = tangents
    s = spike_atan(x)
    sg = grad * (1 / (1 + 10 * x * x))
    return s, sg


# TODO:
# - needs lifted custom JVP to work: https://flax.readthedocs.io/en/latest/design_notes/lift.html
# - check if works
# - order args
# - add correct types
@custom_jvp
def hebbian_surrogate(layer: Type[nn.Module], state, params, input_: Array) -> Tuple[Array, Array]:
    """
    Function that returns surrogate loss, w * x, that results in a Hebbian weight update,
    dw = x * y, when used with loss function L = 1/2 * y^2.

    From "Hebbian learning with gradients...", Miconi et al., arXiv 2021.
    """
    return layer.apply({"params": params, **state}, input_, mutable=["state"])


@hebbian_surrogate.defjvp
def hebbian_surrogate_jvp(primals, tangents):
    layer, state, params, input_, t_pre = primals
    (_,) = tangents
    output, state = hebbian_surrogate(layer, state, params, input_, t_pre)
    wx = state["wx"]
    return (output, state), wx
