import functools
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import jax
from jax import custom_gradient, jit, lax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import torch


DEVICE = "cpu"
jax.config.update("jax_platform_name", DEVICE)

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any

InitFn = Callable[[PRNGKey, Shape, Dtype], Array]
SpikeFn = Callable[[Array], Tuple[Array, Callable[[Array], Array]]]


@custom_gradient
def spike(x: Array) -> Tuple[Array, Callable[[Array], Array]]:
    """
    Spike function (Heaviside) with derivative of arctan surrogate gradient.

    From "Incorporating Learnable Membrane Time Constant...", Fang et al., arXiv 2020.

    TODO:
    - parameterize
    - JIT
    - benchmark thresholding with GPU
    """
    s = lax.gt(x, 0.0).astype(jnp.float32)

    def sg(grad):
        return (grad * (1 / (1 + 10 * x * x)),)

    return s, sg


def constant(scale: float, dtype: Dtype = jnp.float_) -> InitFn:
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return scale * jnp.ones(shape, dtype)

    return init


class LIF(nn.Module):
    size: int
    w_init: InitFn = nn.initializers.lecun_normal()
    leak_i_init: InitFn = constant(0.9)
    leak_v_init: InitFn = constant(0.9)
    thresh_init: InitFn = nn.initializers.ones
    spike_fn: SpikeFn = spike

    @nn.compact
    def __call__(self, input_: Array, state: Array) -> Tuple[Array, Array]:

        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        leak_i = self.param("leak_i", self.leak_i_init, (self.size,))
        leak_v = self.param("leak_v", self.leak_v_init, (self.size,))
        thresh = self.param("thresh", self.thresh_init, (self.size,))

        i, v, s = state

        i = i * leak_i + ff
        v = v * leak_v * (1 - s) + i
        s = self.spike_fn(v - thresh)

        return s, jnp.stack([i, v, s])

    @staticmethod
    def reset_state(state_size: int, shape: Shape) -> Array:
        return jnp.zeros((state_size, *shape))


class Leaky(nn.Module):
    size: int
    leak_init: InitFn = constant(0.9)
    w_init: InitFn = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, input_: Array, state: Array) -> Tuple[Array, Array]:

        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        leak = self.param("leak", self.leak_init, (self.size,))

        out, hidden = state
        hidden = hidden * leak + ff
        out = nn.activation.tanh(hidden)

        return out, jnp.stack([out, hidden])

    @staticmethod
    def reset_state(state_size: int, shape: Shape) -> Array:
        return jnp.zeros((state_size, *shape))


class LIFShapeInf(nn.Module):
    size: int
    w_init: InitFn = nn.initializers.lecun_normal()
    leak_i_init: InitFn = constant(0.9)
    leak_v_init: InitFn = constant(0.9)
    thresh_init: InitFn = nn.initializers.ones
    spike_fn: SpikeFn = spike

    @nn.compact
    def __call__(self, input_: Array, state: Optional[Array] = None) -> Tuple[Array, Array]:

        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        leak_i = self.param("leak_i", self.leak_i_init, (self.size,))
        leak_v = self.param("leak_v", self.leak_v_init, (self.size,))
        thresh = self.param("thresh", self.thresh_init, (self.size,))

        state = state if state is not None else self.reset_state(3, ff.shape)
        i, v, s = state

        i = i * leak_i + ff
        v = v * leak_v * (1 - s) + i
        s = self.spike_fn(v - thresh)

        return s, jnp.stack([i, v, s])

    @staticmethod
    def reset_state(state_size: int, shape: Shape) -> Array:
        return jnp.zeros((state_size, *shape))


class LeakyShapeInf(nn.Module):
    size: int
    leak_init: InitFn = constant(0.9)
    w_init: InitFn = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, input_: Array, state: Optional[Array] = None) -> Tuple[Array, Array]:

        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        leak = self.param("leak", self.leak_init, (self.size,))

        state = state if state is not None else self.reset_state(2, ff.shape)

        out, hidden = state
        hidden = hidden * leak + ff
        out = nn.activation.tanh(hidden)

        return out, jnp.stack([out, hidden])

    @staticmethod
    def reset_state(state_size: int, shape: Shape) -> Array:
        return jnp.zeros((state_size, *shape))


class LIFVar(nn.Module):
    size: int
    w_init: InitFn = nn.initializers.lecun_normal()
    leak_i_init: InitFn = constant(0.9)
    leak_v_init: InitFn = constant(0.9)
    thresh_init: InitFn = nn.initializers.ones
    spike_fn: SpikeFn = spike

    @nn.compact
    def __call__(self, input_: Array) -> Array:

        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        leak_i = self.param("leak_i", self.leak_i_init, (self.size,))
        leak_v = self.param("leak_v", self.leak_v_init, (self.size,))
        thresh = self.param("thresh", self.thresh_init, (self.size,))

        is_init = self.has_variable("state", "i")
        i = self.variable("state", "i", jnp.zeros, ff.shape)
        v = self.variable("state", "v", jnp.zeros, ff.shape)
        s = self.variable("state", "s", jnp.zeros, ff.shape)

        if is_init:
            i.value = i.value * leak_i + ff
            v.value = v.value * leak_v * (1 - s.value) + i.value
            s.value = self.spike_fn(v.value - thresh)

        return s.value


class LeakyVar(nn.Module):
    size: int
    leak_init: InitFn = constant(0.9)
    w_init: InitFn = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, input_: Array) -> Array:

        ff = nn.Dense(self.size, use_bias=False, kernel_init=self.w_init)(input_)
        leak = self.param("leak", self.leak_init, (self.size,))

        is_init = self.has_variable("state", "out")
        out = self.variable("state", "out", jnp.zeros, ff.shape)
        hidden = self.variable("state", "hidden", jnp.zeros, ff.shape)

        if is_init:
            hidden.value = hidden.value * leak + ff
            out.value = nn.activation.tanh(hidden.value)

        return out.value


class BaseNeuron(torch.nn.Module):
    """
    Spiking neuron base class.
    """

    def __init__(self, state_size, fixed_params, learnable_params):
        super().__init__()

        self.state_size = state_size
        self.reset_parameters(fixed_params, learnable_params)

    def forward(self, input_, state):
        state = state if state is not None else self.reset_state(input_)
        return self.activation(input_, state)

    def activation(self, input_, state):
        raise NotImplementedError

    @staticmethod
    def get_spikes(state):
        raise NotImplementedError

    def reset_state(self, input_):
        return torch.zeros(self.state_size, *input_.shape, dtype=input_.dtype, device=input_.device)

    def reset_parameters(self, fixed_params, learnable_params):
        # check if disjoint
        assert fixed_params.keys().isdisjoint(learnable_params.keys()), "A parameter cannot be both fixed and learnable"

        # fixed/non-learnable
        for name, data in fixed_params.items():
            # TODO: what appens if already exists?
            self.register_buffer(name, data)

        # learnable
        for name, data in learnable_params.items():
            # TODO: what appens if already exists?
            setattr(self, name, torch.nn.Parameter(data))


class BaseLIFTorch(BaseNeuron):
    """
    Base LIF neuron with:
    - stateful synaptic current
    - hard reset of membrane potential
    - optionally learnable parameters; either per-neuron or single
    """

    state_size = 3
    neuron_params = ["leak_i", "leak_v", "thresh"]

    def __init__(self, fixed_params, learnable_params, spike_fn):
        super().__init__(3, fixed_params, learnable_params)

        # check parameters are there
        for p in ["leak_i", "leak_v", "thresh"]:
            assert hasattr(self, p), f"{p} not found in {self}"

        # spike mechanism
        self.spike = spike_fn

    # @torch.jit.script
    def activation(self, input_, state):
        # unpack state; spikes always last
        i, v, s = state

        # get parameters
        # TODO: replace with pre-forward hook?
        leak_i, leak_v = self.get_leaks()
        thresh = self.get_thresh()

        # current update: leak, integrate
        i = self.update_i(input_, i, leak_i)

        # voltage update: reset, leak, integrate
        v = self.update_v(i, v, leak_v, s, thresh)

        # spike
        s = self.spike(v - thresh)

        return s, torch.stack([i, v, s])

    @staticmethod
    def get_spikes(state):
        _, _, s = state
        return s

    def get_leaks(self):
        return self.leak_i, self.leak_v

    def get_thresh(self):
        return self.thresh

    def update_i(self, input_, state, leak):
        return state * leak + input_

    def update_v(self, input_, state, leak, reset, thresh):
        return state * leak * (1 - reset) + input_


class BaseSpike(torch.autograd.Function):
    """
    Base spike function without gradient.
    """

    @staticmethod
    def forward(ctx, x, *args):
        ctx.save_for_backward(x, *args)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def spike_torch():
    return BaseSpike.apply


@functools.partial(jit, static_argnums=(0,))
def forward(layer, params, input_, state):
    return layer.apply(params, input_, state)


@functools.partial(jit, static_argnums=(0,))
def forward_statevar(layer, params, input_, state):
    return layer.apply({"params": params, **state}, input_, mutable=["state"])


if __name__ == "__main__":
    print("make sure to run as 'XLA_PYTHON_CLIENT_PREALLOCATE=false python tests/comparison_jax.py'")

    batch = 2
    input_ = random.uniform(random.PRNGKey(0), (batch, 4))

    hidden = 100

    # outside init needs knowledge of shape; can be difficult with convolutions
    leaky = LIF(hidden)
    state = leaky.reset_state(3, (batch, hidden))
    params = leaky.init(random.PRNGKey(0), input_, state)

    # inside init
    leaky_shapeinf = LIFShapeInf(hidden)
    state_shapeinf = None
    params_shapeinf = leaky_shapeinf.init(random.PRNGKey(0), input_, state_shapeinf)

    # state as variable
    leaky_var = LIFVar(hidden)
    state_var, params_var = leaky_var.init(random.PRNGKey(0), input_).pop("params")

    time_start = time.time()
    for _ in range(1000):
        output, state = forward(leaky, params, input_, state)
    print(f"Leaky {DEVICE}: {time.time() - time_start}")

    time_start = time.time()
    for _ in range(1000):
        output_shapeinf, state_shapeinf = forward(leaky_shapeinf, params_shapeinf, input_, state_shapeinf)
    print(f"LeakyShapeInf {DEVICE}: {time.time() - time_start}")

    time_start = time.time()
    for _ in range(1000):
        output_var, state_var = forward_statevar(leaky_var, params_var, input_, state_var)
    print(f"LeakyVar {DEVICE}: {time.time() - time_start}")

    input_ = torch.rand(2, 4)
    learnable_params = dict(leak_i=0.9 * torch.ones(4), leak_v=0.9 * torch.ones(4), thresh=torch.ones(4))
    neuron_torch = BaseLIFTorch({}, learnable_params, spike_torch())
    time1 = time.time()
    state = None
    for _ in range(1000):
        output, state = neuron_torch(input_, state)
    print(f"torch cpu: {time.time() - time1}")

    input_ = torch.rand(2, 4, device="cuda")
    neuron_torch.to("cuda")
    time1 = time.time()
    state = None
    for _ in range(1000):
        output, state = neuron_torch(input_, state)
    print(f"torch gpu: {time.time() - time1}")
