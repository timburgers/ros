"""
Implementation of https://github.com/ThomasMiconi/HebbianCNNPyTorch/blob/main/HebbGrad_Simple_Github.ipynb with spiking neurons and JAX.
Paper: https://arxiv.org/abs/2107.01729
"""

import functools
import sys
import time
from typing import Callable, List

from flax.core.frozen_dict import freeze, unfreeze
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Grayscale, RandomHorizontalFlip

sys.path.append(".")
from spiking.jax.layer.conv import ConvVarTraceLIFVar as ConvLIF
from spiking.jax.utils.dataloading import np_collate, to_numpy
from spiking.jax.utils.typing import Array, Dtype, InitFn


def custom_init(dtype: Dtype = jnp.float_) -> InitFn:
    """
    Initialize with random normal weights with norm 1.
    """

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        sum_dims = tuple(range(1, len(shape)))
        normal = jax.random.normal(key, shape, dtype)
        normalized = normal / jnp.sqrt(jnp.sum(normal ** 2, axis=sum_dims, keepdims=True))
        return normalized

    return init


class ModelLIF(nn.Module):
    """
    Simple fully-convolutional model of leaky-integrate-and-fire neurons.
    """

    @nn.compact
    def __call__(self, input_: Array) -> List[Array]:
        outputs = [None] * 3
        outputs[0] = ConvLIF(100, 5, 2, {}, w_init=custom_init())(input_)
        outputs[1] = ConvLIF(196, 3, 2, {}, w_init=custom_init())(outputs[0])
        outputs[2] = ConvLIF(400, 3, 2, {}, w_init=custom_init())(outputs[1])
        return outputs


def build_dog(size: int) -> Array:
    """
    Build a difference-of-Gaussians kernel.
    """
    kernel1 = np.zeros((size, size))
    kernel1[size // 2, size // 2] = 1
    kernel2 = gaussian_filter(kernel1, sigma=0.5) - gaussian_filter(kernel1, sigma=1.0)
    dog = jnp.asarray(kernel2[..., None, None])  # add dimensions for input and output channel (hwio)
    return dog


def apply_dog(size: int) -> Callable[[Array], Array]:
    """
    Apply a difference-of-Gaussians kernel to the input.
    """
    dog = build_dog(size)

    def apply(input_: Array) -> Array:
        # dimension numbers
        # for JAX, default input is nchw, default kernel is iohw, but we want equality with flax
        # dn = lax.conv_dimension_numbers(input_.shape, dog.shape, ("NHWC", "HWIO", "NHWC"))  # TODO: is this necessary?
        return lax.conv_general_dilated(input_, dog, (1, 1), "SAME", (1, 1), (1, 1), ("NHWC", "HWIO", "NHWC"))

    return apply


def apply_normalize(input_: Array) -> Array:
    """
    Normalize input.
    """
    mean = input_.mean((1, 2, 3), keepdims=True)
    std = input_.std((1, 2, 3), keepdims=True)
    return (input_ - mean) / (1e-10 + std)


@functools.partial(jax.jit, static_argnums=(0,))
def train_step(model, params, state, data):
    print("I'm compiling, this should only print once")

    def hebbian_fn(params, state):
        # forward
        outputs, state = model.apply({"params": params, **state}, data, mutable=["state"])

        assert len(params) == len(state["state"])
        ks = list(state["state"].keys())

        # for all layers (TODO: only those with input and output spiking?)
        for name in ks:
            w = params[name]["ConvPatches_0"]["kernel"]
            pre, post = state["state"][name]["t_pre"], state["state"][name]["t_post"]

            # WTA over channels
            val, _ = lax.top_k(post, k=1)
            valmin = val.min(axis=-1, keepdims=True)
            post = (post >= valmin).astype(jnp.float32) * post

            correlation = jnp.matmul(jnp.expand_dims(pre, axis=-1), jnp.expand_dims(post, axis=-2))
            dw = correlation.mean((0, 1, 2)).reshape(w.shape)

            params = unfreeze(params)
            params[name]["ConvPatches_0"]["kernel"] += 0.001 * dw  # TODO: remove hard-coded LR
            params = freeze(params)

        return outputs, state, params

    outputs, state, params = hebbian_fn(params, state)

    return outputs, state, params


def train_epoch(model, params, state, train_loader):
    # TODO: find a better place for this
    decorrelate = apply_dog(12)

    for data, _ in train_loader:

        # normalize and decorrelate input
        data = decorrelate(apply_normalize(data))

        # run step
        _, state, params = train_step(model, params, state, data)

    return params


def create_train_state(rng, config):
    model = ModelLIF()
    input_shape = (config["batch_size"], 32, 32, 1)

    @jax.jit
    def init(*args):
        return model.init(*args)

    state, params = init(rng, jnp.ones(input_shape)).pop("params")
    return model, params, state


def main(config):
    mp.set_start_method("spawn")

    rng = jax.random.PRNGKey(config["seed"])
    torch.manual_seed(config["seed"])

    transforms = Compose([Grayscale(), RandomHorizontalFlip(), to_numpy])
    train_ds = CIFAR10(root="data", train=True, download=True, transform=transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=np_collate,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )

    rng, init_rng = jax.random.split(rng)
    model, params, state = create_train_state(init_rng, config)

    for e in range(config["epochs"]):
        e_start = time.time()
        # TODO: state is now reset every epoch, is this ok?
        params = train_epoch(model, params, state, train_loader)
        print(f"epoch {e}: {time.time() - e_start:.2f}s")

        # plot
        kernel0 = params["ConvTraceLIFVar_0"]["ConvPatches_0"]["kernel"]
        height, width, inc, outc = kernel0.shape
        image = jnp.transpose(kernel0, (3, 0, 1, 2))
        cols = 10
        image = jnp.reshape(image, (image.shape[0] // cols, cols, height, width, inc))
        image = jnp.reshape(image, (image.shape[0] * height, image.shape[1] * width, inc))
        plt.figure()
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    batch_size = 100
    config = dict(
        batch_size=batch_size,
        learning_rate=0.01 / batch_size,
        epochs=10,
        seed=0,
        log_dir="logs",
    )

    main(config)
