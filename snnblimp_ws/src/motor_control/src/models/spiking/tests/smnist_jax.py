from argparse import ArgumentParser
import functools
import sys
import time
from typing import List, Tuple

import flax.linen as nn
from flax.metrics.tensorboard import SummaryWriter
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from torchvision.datasets import MNIST
from torchvision.transforms import Compose

sys.path.append(".")
from spiking.jax.layer.linear import LinearLIF
from spiking.jax.utils.dataloading import np_collate, to_numpy
from spiking.jax.utils.typing import Array, Shape


class ScanModel(nn.Module):
    @functools.partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, state: List[Array], input_: Array) -> Tuple[List[Array], Array]:
        state[0], output = LinearLIF(100, {})(state[0], input_)
        output = nn.Dense(10, use_bias=False)(output)
        return state, output

    @staticmethod
    def reset_state(input_shape: Shape) -> List[Array]:
        state = [None] * 1
        n, _, _ = input_shape
        state[0] = jnp.zeros((3, n, 100))
        return state


def cross_entropy_loss(*, logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


def create_train_state(rng, config):
    model = ScanModel()
    input_shape = (config["batch_size"], 784, 1)

    @jax.jit  # jit init; faster and state doesn't change
    def init(*args):
        return model.init(*args)

    state = model.reset_state(input_shape)
    params = init(rng, state, jnp.ones(input_shape))
    tx = optax.sgd(config["learning_rate"], config["momentum"])
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), state


@jax.jit
def train_step(train_state, state, data, target):
    print("I'm compiling, and this should only print once!")

    def loss_fn(params, state):
        state, logits = train_state.apply_fn(params, state, data)
        loss = cross_entropy_loss(logits=logits[:, -1, :], labels=target)  # predict at end of sequence
        return loss, logits[:, -1, :]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(train_state.params, state)
    train_state = train_state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=target)
    return train_state, metrics


def train_epoch(train_state, state, train_loader):
    batch_metrics = []
    for data, target in train_loader:
        train_state, metrics = train_step(train_state, state, data, target)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

    return train_state, epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


@jax.jit
def eval_step(train_state, state, data, target):
    _, logits = train_state.apply_fn(train_state.params, state, data)
    return compute_metrics(logits=logits[:, -1, :], labels=target)


def eval_model(train_state, state, test_loader):
    batch_metrics = []
    for data, target in test_loader:
        metrics = eval_step(train_state, state, data, target)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

    # TODO: why use this below?
    # metrics = jax.device_get(metrics)
    # summary = jax.tree_map(lambda x: x.item(), metrics)

    return epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


def to_flat(img):
    return img.reshape(-1, 1) * 100


def train_and_evaluate(config: dict, workdir: str, debug: bool) -> TrainState:
    start_time = time.time()
    multiprocessing.set_start_method("spawn")

    # seeding
    rng = jax.random.PRNGKey(0)
    torch.manual_seed(0)

    train_ds = MNIST("data", download=True, train=True, transform=Compose([to_numpy, to_flat]))
    test_ds = MNIST("data", download=True, train=False, transform=Compose([to_numpy, to_flat]))
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
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=np_collate,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )

    # logging
    if not debug:
        summary_writer = SummaryWriter(workdir)
        summary_writer.hparams(config)

    rng, init_rng = jax.random.split(rng)
    train_state, state = create_train_state(init_rng, config)

    startup_time = time.time()
    print(f"startup time: {startup_time - start_time:.2f}s")

    for epoch in range(1, config["num_epochs"] + 1):
        loop_time = time.time()

        train_state, train_loss, train_accuracy = train_epoch(train_state, state, train_loader)
        train_time = time.time()

        test_loss, test_accuracy = eval_model(train_state, state, test_loader)
        eval_time = time.time()

        print(
            f"epoch:{epoch:3}, time (train/eval): {train_time - loop_time:.2f}s / {eval_time - train_time:.2f}s, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy * 100:.2f}, test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy * 100:.2f}"
        )

        if not debug:
            summary_writer.scalar("train_loss", train_loss, epoch)
            summary_writer.scalar("train_accuracy", train_accuracy, epoch)
            summary_writer.scalar("test_loss", test_loss, epoch)
            summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    if not debug:
        summary_writer.flush()
        summary_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = dict(
        learning_rate=0.001,
        momentum=0.9,
        batch_size=1024,
        num_epochs=20,
    )

    train_and_evaluate(config, "./logs", args.debug)
