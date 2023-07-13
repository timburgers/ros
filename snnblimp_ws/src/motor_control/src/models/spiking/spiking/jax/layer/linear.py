from typing import Any, Dict, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp

import spiking.jax.layer.base as base
import spiking.jax.neuron.lif as lif
from spiking.jax.utils.initializations import constant
from spiking.jax.utils.typing import Array, InitFn, Shape


class Linear(base.Layer):
    """
    Base densely-connected feedforward linear layer with:
    - no bias
    - weights initialized with Lecun truncated normal (fan-in)

    TODO:
    - check weight init
    - add defaults to neuron_params, neuron_model
    """

    size: int
    neuron_params: Dict[str, Any]
    neuron_model: Type[nn.Module]
    w_init: InitFn = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, state: Array, input_: Array) -> Tuple[Array, Array]:
        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        state, output = self.neuron_model(self.size, **self.neuron_params)(state, ff)
        return state, output

    # TODO: add shape inference
    def reset_state(self, shape: Shape) -> Array:
        return self.neuron_model.reset_state(shape)


class LinearVar(base.Layer):
    """
    Base densely-connected feedforward linear layer with:
    - no bias
    - weights initialized with Lecun truncated normal (fan-in)

    TODO:
    - check weight init
    - add defaults to neuron_params, neuron_model
    """

    size: int
    neuron_params: Dict[str, Any]
    neuron_model: Type[nn.Module]
    w_init: InitFn = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        output = self.neuron_model(self.size, **self.neuron_params)(ff)
        return output


class LinearVarTrace(LinearVar, base.Layer):
    """
    Linear layer with trace recording and state as variable.

    TODO:
    - by putting both pre and post trace in a layer, we duplicate traces
    - it might be better to have trace as state in a neuron (no duplicates)
    - but will it be as simple to implement Hebbian learning then?
    """

    leak_t_pre_init: InitFn = constant(0.9)
    leak_t_post_init: InitFn = constant(0.9)

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        output = self.neuron_model(self.size, **self.neuron_params)(ff)

        in_size = input_.shape[-1]
        leak_t_pre = self.param("leak_t_pre", self.leak_t_pre_init, (in_size,))
        leak_t_post = self.param("leak_t_post", self.leak_t_post_init, (self.size,))

        is_init = self.has_variable("state", "t_pre")
        t_pre = self.variable("state", "t_pre", jnp.zeros, input_.shape)
        t_post = self.variable("state", "t_post", jnp.zeros, ff.shape)

        if is_init:
            # TODO: patches of input or patches of pre trace?
            # regular neuron update function (leak, add)
            t_pre.value = self.neuron_model.update(t_pre.value, input_, leak_t_pre)
            t_post.value = self.neuron_model.update(t_post.value, output, leak_t_post)

        return output


class RecurrentLinear(base.Layer):
    """
    Base densely-connected recurrent linear layer with:
    - no bias
    - weights initialized with Lecun truncated normal (fan-in)

    TODO:
    - check weight init
    - add defaults to neuron_params, neuron_model
    """

    size: int
    neuron_params: Dict[str, Any]
    neuron_model: Type[nn.Module]
    w_init_ff: InitFn = nn.initializers.lecun_normal()
    w_init_rec: InitFn = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, state: Array, input_: Array) -> Tuple[Array, Array]:
        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init_ff)(input_)

        s = self.neuron_model.get_output(state)
        rec = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init_rec)(s)

        state, output = self.neuron_model(self.size, **self.neuron_params)(state, ff + rec)
        return state, output

    # TODO: add shape inference
    def reset_state(self, shape: Shape) -> Array:
        return self.neuron_model.reset_state(shape)


class RecurrentLinearVar(base.Layer):
    """
    Base densely-connected recurrent linear layer with:
    - no bias
    - weights initialized with Lecun truncated normal (fan-in)

    TODO:
    - is there a penalty to doing things like this, with every time jnp.zeros_like?
    - does this make sense? can we do without separate setup?
    - inherit from BaseLinear?
    - check weight init
    - add defaults to neuron_params, neuron_model
    """

    size: int
    neuron_params: Dict[str, Any]
    neuron_model: Type[nn.Module]
    w_init_ff: InitFn = nn.initializers.lecun_normal()
    w_init_rec: InitFn = nn.initializers.lecun_normal()

    # separate setup: we need to check neuron vars before forwarding it
    def setup(self):
        self.ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init_ff)
        self.rec = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init_rec)
        self.neuron = self.neuron_model(self.size, **self.neuron_params)

    def __call__(self, input_: Array) -> Array:
        ff = self.ff(input_)
        rec = self.rec(self.neuron.get_output(ff))
        output = self.neuron(ff + rec)
        return output


class LinearLIF(Linear):
    """
    Linear layer with LIF activation.
    """

    neuron_model: Type[nn.Module] = lif.Neuron


class LinearLIFVar(LinearVar):
    """
    Linear LIF layer with state as variable.
    """

    neuron_model: Type[nn.Module] = lif.NeuronVar


class RecurrentLinearLIF(RecurrentLinear):
    """
    Recurrent linear layer with LIF activation.
    """

    neuron_model: Type[nn.Module] = lif.Neuron


class RecurrentLinearLIFVar(RecurrentLinearVar):
    """
    Recurrent linear LIF layer with state as variable.
    """

    neuron_model: Type[nn.Module] = lif.NeuronVar
