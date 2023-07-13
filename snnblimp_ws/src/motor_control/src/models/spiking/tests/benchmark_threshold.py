import time
import random
import statistics

import torch
import jax
import jax.numpy as np


def gt_torch(x):
    return x.gt(0).float()


@jax.jit
def gt_jax(x):
    return (x > 0).astype(np.float32)


@jax.jit
def gt_jax2(x):
    return np.greater(x, 0).astype(np.float32)


@jax.jit
def gt_jax3(x):
    return jax.lax.gt(x, 0.0).astype(np.float32)


functions = gt_jax, gt_jax2, gt_jax3
# functions = gt_torch,
times = {f.__name__: [] for f in functions}
runs = 10000
inputs = jax.random.normal(jax.random.PRNGKey(0), (runs, 100, 100))
# inputs = torch.randn(runs, 100, 100)

for i in range(runs):
    func = random.choice(functions)
    t0 = time.time()
    func(inputs[i])
    t1 = time.time()
    times[func.__name__].append((t1 - t0) * 1000)

for name, numbers in times.items():
    print("FUNCTION:", name, "Used", len(numbers), "times")
    print("\tMEDIAN", statistics.median(numbers))
    print("\tMEAN  ", statistics.mean(numbers))
    print("\tSTDEV ", statistics.stdev(numbers))
