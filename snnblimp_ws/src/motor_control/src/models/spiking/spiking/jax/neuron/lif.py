from typing import Tuple

import jax.numpy as jnp
import flax.linen as nn

import spiking.jax.neuron.base as base
from spiking.jax.utils.initializations import constant
from spiking.jax.utils.surrogates import spike_atan
from spiking.jax.utils.typing import Array, InitFn, Shape, SpikeFn


class Neuron(base.Neuron):
    """
    Leaky-integrate-and-fire neuron with:
    - stateful synaptic current
    - hard reset of membrane potential
    - per-neuron learnable parameters
    """

    leak_i_init: InitFn = constant(0.9)
    leak_v_init: InitFn = constant(0.9)
    thresh_init: InitFn = nn.initializers.ones
    spike_fn: SpikeFn = spike_atan

    @nn.compact
    def __call__(self, state: Array, input_: Array) -> Tuple[Array, Array]:
        leak_i = self.param("leak_i", self.leak_i_init, (self.size,))
        leak_v = self.param("leak_v", self.leak_v_init, (self.size,))
        thresh = self.param("thresh", self.thresh_init, (self.size,))

        i, v, s = state

        i = self.update(i, input_, leak_i)
        v = self.update_reset(v, i, leak_v, s, thresh)
        s = self.spike_fn(v - thresh)

        return jnp.stack([i, v, s]), s

    @staticmethod
    def reset_state(shape: Shape) -> Array:
        return jnp.zeros((3, *shape))

    @staticmethod
    def get_output(state: Array) -> Array:
        _, _, s = state
        return s


class NeuronVar(base.Neuron):
    """
    LIF neuron with state as variable:
    - easy shape inference
    - slight performance improvement
    """

    leak_i_init: InitFn = constant(0.9)
    leak_v_init: InitFn = constant(0.9)
    thresh_init: InitFn = nn.initializers.ones
    spike_fn: SpikeFn = spike_atan

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        leak_i = self.param("leak_i", self.leak_i_init, (self.size,))
        leak_v = self.param("leak_v", self.leak_v_init, (self.size,))
        thresh = self.param("thresh", self.thresh_init, (self.size,))

        is_init = self.has_variable("state", "i")
        i = self.variable("state", "i", jnp.zeros, input_.shape)
        v = self.variable("state", "v", jnp.zeros, input_.shape)
        s = self.variable("state", "s", jnp.zeros, input_.shape)

        if is_init:
            i.value = self.update(i.value, input_, leak_i)
            v.value = self.update_reset(v.value, i.value, leak_v, s.value, thresh)
            s.value = self.spike_fn(v.value - thresh)

        return s.value

    def get_output(self, default_like: Array) -> Array:
        # calling jnp.zeros_like here saves subsequent calls
        return self.variables.get("state", dict()).get("s", jnp.zeros_like(default_like))
