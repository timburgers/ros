import functools
import sys
import time
from typing import Tuple

import flax.linen as jnn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParametersJIT,
    lif_feed_forward_step,
    _lif_feed_forward_step_jit,
    lif_feed_forward_step_sparse,
)
from norse.torch.module.encode import PoissonEncoder

sys.path.append(".")
from spiking.jax.layer.linear import LinearLIFVar
from spiking.jax.utils.typing import Array


# DEVICE = "cpu"
# jax.config.update("jax_platform_name", DEVICE)


def init_lif_params(*shape):
    params = dict(
        leak_i=torch.ones(shape) * 0.9,
        leak_v=torch.ones(shape) * 0.9,
        thresh=torch.ones(shape),
    )
    return params


class NorseLIF_ffstep(torch.jit.ScriptModule):
    def __init__(self, layer_size):
        super().__init__()
        self.ff = tnn.Linear(layer_size, layer_size, bias=False)

    @torch.jit.script_method
    def forward(
        self, state: LIFFeedForwardState, input_: torch.Tensor, p: LIFParametersJIT
    ) -> Tuple[LIFFeedForwardState, torch.Tensor]:
        ff = self.ff(input_)
        # s, state = lif_feed_forward_step(ff, state, p=p, dt=1.0)
        s, state = _lif_feed_forward_step_jit(ff, state, p=p, dt=1.0)
        return state, s


# class NorseLIF_ffstep_inside(tnn.Module):
class NorseLIF_ffstep_inside(torch.jit.ScriptModule):
    def __init__(self, layer_size):
        super().__init__()
        self.ff = tnn.Linear(layer_size, layer_size, bias=False)

    @torch.jit.script_method
    def forward(
        self, state: LIFFeedForwardState, input_: torch.Tensor, p: LIFParametersJIT
    ) -> Tuple[LIFFeedForwardState, torch.Tensor]:
        outputs = []  # difference with empty tensor and indexing = negligible

        for inp in input_:
            # TODO: lif_feed_forward_step?
            ff = self.ff(inp)
            # s, state = lif_feed_forward_step(ff, state, p=p, dt=1.0)  # works, but then dont decorate with @torch.jit.script_method
            s, state = _lif_feed_forward_step_jit(ff, state, p=p, dt=1.0)
            outputs.append(s)

        return state, torch.stack(outputs)


# TODO: add JIT version, add sparse version


class JaxWrapLIF(jnn.Module):
    layer_size: int

    @jnn.compact
    def __call__(self, input_: Array) -> Array:
        output = LinearLIFVar(self.layer_size, {})(input_)
        return output


def benchmark_norse_lif_inside(batch_size, seq_len, layer_size, device, label):
    with torch.no_grad():
        # create input
        # spike chance: 0.1 * 1 * 0.3
        # TODO: parameterize outside
        input_ = PoissonEncoder(seq_len, f_max=0.1, dt=1.0)(
            0.3 * torch.ones(batch_size, layer_size, device=device)
        ).contiguous()

        # create model
        model = NorseLIF_ffstep_inside(layer_size)
        # model = NorseLIF_ffstep(layer_size)
        model.to(device)

        # params and initial state
        p = LIFParametersJIT(
            tau_syn_inv=torch.full((layer_size,), 1 - 0.9, device=device),
            tau_mem_inv=torch.full((layer_size,), 1 - 0.9, device=device),
            v_leak=torch.as_tensor(0.0),
            v_th=torch.ones(layer_size, device=device),
            v_reset=torch.as_tensor(0.0),
            method="super",
            alpha=torch.as_tensor(10.0),
        )
        state = LIFFeedForwardState(
            v=p.v_leak,
            i=torch.zeros(batch_size, layer_size, device=device),
        )

        # run
        start = time.time()

        # approach 1
        model(state, input_, p)

        # approach 2: 2x slower for Norse + JIT
        # outputs = []
        # for inp in input_:
        #     state, s = model(state, inp, p)
        #     outputs.append(s)
        # outputs = torch.stack(outputs)

        torch.cuda.synchronize()
        end = time.time()

        duration = end - start
        print(f"{label}: {duration}")

        return duration


def benchmark_jax_lif_inside(batch_size, seq_len, layer_size, device, label):
    # create input
    # spike chance: 0.1 * 1 * 0.3
    # TODO: parameterize outside
    input_ = PoissonEncoder(seq_len, f_max=0.1, dt=1.0)(0.3 * torch.ones(batch_size, layer_size)).contiguous()
    input_ = np.asarray(input_, dtype=jnp.float32)  # TODO: or jnp?

    # create model
    rng = jax.random.PRNGKey(0)
    model = JaxWrapLIF(layer_size)  # TODO: needs wrapper module to prevent "unhashable attributes" (LIFVar) error

    @jax.jit
    def init(*args):
        return model.init(*args)

    input_shape = (batch_size, layer_size)
    state, params = init(rng, jnp.ones(input_shape)).pop("params")

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(model, state, params, input_):
        s, state = model.apply({"params": params, **state}, input_, mutable=["state"])
        return state, s

    @functools.partial(jax.jit, static_argnums=(0,))
    def run(model, state, params, input_):
        outputs = []

        for inp in input_:
            s, state = model.apply({"params": params, **state}, inp, mutable=["state"])
            outputs.append(s)

        return state, jnp.stack(outputs)

    # pre-heat jit
    _ = step(model, state, params, input_[0])

    # run
    start = time.time()

    # approach 1: much much slower
    # run(model, state, params, input_)

    # approach 2
    outputs = []
    for inp in input_:
        state, s = step(model, state, params, inp)
        outputs.append(s)
    outputs = jnp.stack(outputs).block_until_ready()

    end = time.time()

    duration = end - start
    print(f"{label}: {duration}")

    return duration


if __name__ == "__main__":
    # benchmark
    runs = 10
    batch_size = 32
    seq_len = 500
    layer_size = 5000
    device = torch.device("cuda")
    data = {}

    # run
    for _ in range(runs):
        benchmark_norse_lif_inside(batch_size, seq_len, layer_size, device, "norse_lif_inside")

    for _ in range(runs):
        benchmark_jax_lif_inside(batch_size, seq_len, layer_size, device, "jax_lif_inside")
