from argparse import ArgumentParser
import sys
import time

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
from spiking.jax.layer.linear import RecurrentLinearLIFVar
from spiking.jax.utils.dataloading import np_collate, to_numpy
from spiking.jax.utils.typing import Array


class Model(nn.Module):
    @nn.compact
    def __call__(self, input_: Array) -> Array:
        output = RecurrentLinearLIFVar(100, {})(input_)
        output = RecurrentLinearLIFVar(100, {})(output)
        output = nn.Dense(10)(output)
        return output


def cross_entropy_loss(*, logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


def create_train_state(rng, config):
    model = Model()
    input_shape = (config["batch_size"], 1)

    @jax.jit
    def init(*args):
        return model.init(*args)

    state, params = init(rng, jnp.ones(input_shape)).pop("params")
    # tx = optax.sgd(config["learning_rate"], config["momentum"])
    tx = optax.adam(config["learning_rate"])
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), state


@jax.jit
def train_step(train_state, state, data, target):
    print("I'm compiling, and this should only print once!")

    def loss_fn(params, state):
        def scan_model(state, input_):
            # mind that state from variables comes second
            output, state = train_state.apply_fn({"params": params, **state}, input_, mutable=["state"])
            return state, output

        state, logits = jax.lax.scan(scan_model, state, jnp.swapaxes(data, 0, 1))
        loss = cross_entropy_loss(logits=logits[-1, :, :], labels=target)
        return loss, logits[-1, :, :]

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
    def scan_model(state, input_):
        output, state = train_state.apply_fn({"params": train_state.params, **state}, input_, mutable=["state"])
        return state, output

    _, logits = jax.lax.scan(scan_model, state, jnp.swapaxes(data, 0, 1))
    return compute_metrics(logits=logits[-1, :, :], labels=target)


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
    return img.reshape(-1, 1)


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
