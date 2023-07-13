import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

sys.path.append(".")
from spiking.jax.utils.dataloading import np_collate, to_numpy


def jax_transform(input_):
    return jnp.expand_dims(
        jnp.asarray(input_, dtype=jnp.float32) / 255, -1
    )  # jax/flax uses nhwc instead of nchw (torch)


def jax_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(jax_collate(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


def dataloading_torch(batch_size, num_workers, pin_memory, persistent_workers, device):
    startup_time = time.time()

    train_ds = MNIST("data", download=True, train=True, transform=Compose([ToTensor()]))
    test_ds = MNIST("data", download=True, train=False, transform=Compose([ToTensor()]))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    start_time = time.time()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

    finish_time = time.time()

    print(
        f"torch took {finish_time - startup_time:.2f}s; ({start_time - startup_time:.2f}s/{finish_time - start_time:.2f}s) (startup/loading)"
    )


def dataloading_jax(batch_size, num_workers, pin_memory, persistent_workers, device):
    startup_time = time.time()

    train_ds = MNIST("data", download=True, train=True, transform=jax_transform)
    test_ds = MNIST("data", download=True, train=False, transform=jax_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=jax_collate,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=jax_collate,
        pin_memory=pin_memory,
    )

    start_time = time.time()

    for i, (x, y) in enumerate(train_loader):
        x, y = jax.device_put(x), jax.device_put(y)  # don't think this is necesary, but good for fair timing

    for i, (x, y) in enumerate(test_loader):
        x, y = jax.device_put(x), jax.device_put(y)

    finish_time = time.time()

    print(
        f"jax took {finish_time - startup_time:.2f}s; ({start_time - startup_time:.2f}s/{finish_time - start_time:.2f}s) (startup/loading)"
    )


def dataloading_numpy(batch_size, num_workers, pin_memory, persistent_workers, device):
    startup_time = time.time()

    train_ds = MNIST("data", download=True, train=True, transform=to_numpy)
    test_ds = MNIST("data", download=True, train=False, transform=to_numpy)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=np_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=np_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    start_time = time.time()

    for i, (x, y) in enumerate(train_loader):
        x, y = jax.device_put(x), jax.device_put(y)  # don't think this is necesary, but good for fair timing

    for i, (x, y) in enumerate(test_loader):
        x, y = jax.device_put(x), jax.device_put(y)

    finish_time = time.time()

    print(
        f"numpy took {finish_time - startup_time:.2f}s; ({start_time - startup_time:.2f}s/{finish_time - start_time:.2f}s) (startup/loading)"
    )


if __name__ == "__main__":
    print("make sure to run as 'XLA_PYTHON_CLIENT_PREALLOCATE=false python tests/dataloading.py'")

    repeats = 5
    mp_method = "fork"
    batch_size = 1024
    num_workers = 16
    pin_memory = True
    persistent_workers = True
    device = torch.device("cuda")

    multiprocessing.set_start_method(mp_method)

    # functions = [dataloading_torch, dataloading_jax, dataloading_numpy]
    functions = [dataloading_numpy]
    for func in functions:
        for _ in range(repeats):
            func(batch_size, num_workers, pin_memory, persistent_workers, device)
