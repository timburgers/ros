import functools
import time
from typing import Any

import flax.linen as jnn
import jax
import jax.numpy as jnp
import jax.experimental.stax as stax
import numpy as np
import torch
import torch.nn as tnn


Array = Any


class TorchLinear(tnn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.ff = tnn.Linear(layer_size, layer_size, bias=False)

    def forward(self, input_):
        output = self.ff(input_)
        return output


class JaxLinear(jnn.Module):
    layer_size: int

    @jnn.compact
    def __call__(self, input_: Array) -> Array:
        output = jnn.Dense(self.layer_size, use_bias=False)(input_)
        return output


def benchmark_torch_linear(inputs, layer_size, device, label):
    with torch.no_grad():
        # create model
        model = TorchLinear(layer_size)
        model.to(device)

        # start_cuda = torch.cuda.Event(enable_timing=True)
        # end_cuda = torch.cuda.Event(enable_timing=True)

        # run
        start = time.time()
        # start_cuda.record()

        outputs = []
        for input_ in inputs:
            output = model(input_)
            outputs.append(output)
        outputs = torch.stack(outputs)
        torch.cuda.synchronize()

        # end_cuda.record()
        end = time.time()

        duration = end - start
        print(f"{label}: {duration}")
        # print(start_cuda.elapsed_time(end_cuda) / 1000)

        return duration


def benchmark_jax_linear(inputs, layer_size, device, label):
    # create model
    rng = jax.random.PRNGKey(0)
    model = JaxLinear(layer_size)

    @jax.jit
    def init(*args):
        return model.init(*args)

    input_shape = (batch_size, layer_size)
    params = init(rng, jnp.ones(input_shape))

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(model, params, input_):
        output = model.apply(params, input_)
        return output

    # pre-heat jit
    _ = step(model, params, inputs[0])

    # run
    start = time.time()

    outputs = []
    for input_ in inputs:
        output = step(model, params, input_)
        outputs.append(output)
    outputs = jnp.stack(outputs)
    outputs[0].block_until_ready()

    end = time.time()

    duration = end - start
    print(f"{label}: {duration}")

    return duration


def benchmark_stax_linear(inputs, layer_size, device, label):
    # create model
    rng = jax.random.PRNGKey(0)
    init_fn, apply_fn = stax.Dense(layer_size)  # TODO: has to have bias

    input_shape = (batch_size, layer_size)
    _, params = init_fn(rng, input_shape)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(apply_fn, params, input_):
        output = apply_fn(params, input_)
        return output

    # pre-heat jit
    _ = step(apply_fn, params, inputs[0])

    # run
    start = time.time()

    outputs = []
    for input_ in inputs:
        output = step(apply_fn, params, input_)
        outputs.append(output)
    outputs = jnp.stack(outputs)
    outputs[0].block_until_ready()

    end = time.time()

    duration = end - start
    print(f"{label}: {duration}")

    return duration


if __name__ == "__main__":
    # benchmark
    # outcome: JAX is competitive/faster than torch for larger matrices
    # due to relatively high dispatch cost
    # https://github.com/google/jax/discussions/8497
    runs = 10
    batch_size = 32
    seq_len = 500
    layer_size = 5000
    device = torch.device("cuda")

    # input data
    input_torch = (torch.rand(seq_len, batch_size, layer_size, device=device) < 0.2).float().contiguous()
    input_np = np.asarray(input_torch.cpu(), dtype=jnp.float32)
    input_jnp = jnp.asarray(input_torch.cpu())

    # pytorch / loop outside module
    for _ in range(runs):
        benchmark_torch_linear(input_torch, layer_size, device, "torch")

    print()

    # jax / jnp input / loop outside module
    for _ in range(runs):
        benchmark_jax_linear(input_jnp, layer_size, device, "flax, jnp input")

    print()

    # stax / jnp input / loop outside module
    for _ in range(runs):
        benchmark_stax_linear(input_jnp, layer_size, device, "stax, jnp input")
