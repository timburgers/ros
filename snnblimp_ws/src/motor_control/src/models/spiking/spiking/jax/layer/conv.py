from typing import Any, Dict, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

import spiking.jax.layer.base as base
import spiking.jax.neuron.lif as lif
from spiking.jax.synapse.conv import ConvPatches
from spiking.jax.utils.initializations import constant
from spiking.jax.utils.typing import Array, InitFn, Shape


class Conv(base.Layer):
    """
    Base convolutional feedforward layer with:
    - no bias
    - weights initialized with Lecun truncated normal (fan-in)

    TODO:
    - check weight init
    - add defaults to stride, neuron_params, neuron_model
    """

    channels: int
    kernel_size: int
    stride: int
    neuron_params: Dict[str, Any]
    neuron_model: Type[nn.Module]
    w_init: InitFn = nn.initializers.lecun_normal()  # TODO: does this consider entire receptive field?

    @nn.compact
    def __call__(self, state: Array, input_: Array) -> Tuple[Array, Array]:
        ff = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding="VALID",  # pytorch default is 0 = valid
            use_bias=False,
            kernel_init=self.w_init,
        )(input_)
        state, output = self.neuron_model(self.channels, **self.neuron_params)(state, ff)
        return state, output

    # TODO: add shape inference
    def reset_state(self, shape: Shape) -> Array:
        return self.neuron_model.reset_state(shape)


class ConvVar(base.Layer):
    """
    Base convolutional feedforward layer with:
    - no bias
    - weights initialized with Lecun truncated normal (fan-in)

    TODO:
    - check weight init
    - add defaults to stride, neuron_params, neuron_model
    """

    channels: int
    kernel_size: int
    stride: int
    neuron_params: Dict[str, Any]
    neuron_model: Type[nn.Module]
    w_init: InitFn = nn.initializers.lecun_normal()  # TODO: does this consider entire receptive field?

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        ff = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding="VALID",  # pytorch default is 0 = valid
            use_bias=False,
            kernel_init=self.w_init,
        )(input_)
        output = self.neuron_model(self.channels, **self.neuron_params)(ff)
        return output


class ConvVarTrace(ConvVar, base.Layer):
    """
    Convolutional layer with trace recording and state as variable.

    TODO:
    - by putting both pre and post trace in a layer, we duplicate traces
    - it might be better to have trace as state in a neuron (no duplicates)
    - but will it be as simple to implement Hebbian learning then?
    """

    leak_t_pre_init: InitFn = constant(0.9)
    leak_t_post_init: InitFn = constant(0.9)

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        ff, input_patches = ConvPatches(
            self.channels,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding="VALID",  # pytorch default is 0 = valid
            use_bias=False,
            kernel_init=self.w_init,
        )(input_)
        output = self.neuron_model(self.channels, **self.neuron_params)(ff)

        in_size = input_patches.shape[-1]  # TODO: is this correct? we now have way more neurons than channels
        leak_t_pre = self.param("leak_t_pre", self.leak_t_pre_init, (in_size,))
        leak_t_post = self.param("leak_t_post", self.leak_t_post_init, (self.channels,))

        is_init = self.has_variable("state", "t_pre")
        t_pre = self.variable("state", "t_pre", jnp.zeros, input_patches.shape)
        t_post = self.variable("state", "t_post", jnp.zeros, ff.shape)

        if is_init:
            # TODO: patches of input or patches of pre trace?
            # regular neuron update function (leak, add)
            t_pre.value = self.neuron_model.update(t_pre.value, input_patches, leak_t_pre)
            t_post.value = self.neuron_model.update(t_post.value, output, leak_t_post)

        return output


class ConvLIF(Conv):
    """
    Convolutional layer with LIF activation.
    """

    neuron_model: Type[nn.Module] = lif.Neuron


class ConvLIFVar(ConvVar):
    """
    Convolutional LIF layer with state as variable.
    """

    neuron_model: Type[nn.Module] = lif.NeuronVar


class ConvVarTraceLIFVar(ConvVarTrace):
    """
    Convolutional LIF layer with trace recording and state as variable.
    """

    neuron_model: Type[nn.Module] = lif.NeuronVar
