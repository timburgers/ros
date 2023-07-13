import functools
import sys
import time
from typing import List, Tuple

import flax.linen as nn
from flax.metrics.tensorboard import SummaryWriter  # TODO: not using tensorflow prevents memory errors after each epoch

# TODO: tensorflow also preallocates: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth, this solves errors
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from torchvision.datasets import MNIST

sys.path.append(".")
from spiking.jax.layer.conv import ConvLIF, ConvLIFVar
from spiking.jax.layer.linear import LinearLIF, LinearLIFVar
from spiking.jax.utils.dataloading import np_collate, to_numpy
from spiking.jax.utils.typing import Array, Shape


class Stateless(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


class Model(nn.Module):
    """
    Simple convolutional spiking model.
    """

    @nn.compact
    def __call__(self, state: List[Array], input_: Array) -> Tuple[List[Array], Array]:
        state[0], output = ConvLIF(32, 3, 2, {})(state[0], input_)
        state[1], output = ConvLIF(64, 3, 2, {})(state[1], output)
        output = output.reshape(output.shape[0], -1)  # flatten
        state[2], output = LinearLIF(256, {})(state[2], output)
        output = nn.Dense(10)(output)
        # output = nn.log_softmax(output)
        return state, output

    @staticmethod
    def reset_state(input_shape: Shape) -> List[Array]:
        state = [None] * 3
        n, h, w, _ = input_shape
        state[0] = jnp.zeros((3, n, h // 2 - 1, w // 2 - 1, 32))
        state[1] = jnp.zeros((3, n, h // 4 - 1, w // 4 - 1, 64))
        state[2] = jnp.zeros((3, n, 256))
        return state


class ModelScan(nn.Module):
    """
    Scan wrapper around model.
    """

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        state = Model.reset_state(input_.shape)
        model = nn.scan(
            Model, variable_broadcast="params", split_rngs={"params": False}, in_axes=1, out_axes=1
        )  # TODO: how can we do this without duplicating data?
        return model()(input_, state)  # TODO: I think this should be other way around; state first?


class ModelVar(nn.Module):
    """
    Simple convolutional spiking model with state as variable (allowing shape inference).

    TODO:
    - also seems to speed up on GPU? validate!
    """

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        output = ConvLIFVar(32, 3, 2, {})(input_)
        output = ConvLIFVar(64, 3, 2, {})(output)
        output = output.reshape(output.shape[0], -1)  # flatten
        output = LinearLIFVar(256, {})(output)
        output = nn.Dense(10)(output)
        # output = nn.log_softmax(output)
        return output

    @staticmethod
    def reset_state(input_shape: Shape):
        # TODO: take in state and set to 0?
        pass


# TODO: what does this * do? is that why we use keywords later?
def cross_entropy_loss(*, logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))


# @jax.vmap
# def cross_entropy_loss_vmap(*, logits, labels):
#     one_hot = jax.nn.one_hot(labels, num_classes=10)
#     return optax.softmax_cross_entropy(logits, one_hot)


# TODO: what does this * do? is that why we use keywords later?
def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    # loss = cross_entropy_loss_vmap(logits=logits, labels=labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics


def create_train_state(rng, config):
    model = Model()
    input_shape = (config.batch_size, 28, 28, 1)
    state = model.reset_state(input_shape)  # TODO: move inside model? penalty?

    # jitting init doesn't modify state
    @jax.jit  # TODO: this makes a big difference for startup; 5 -> 2 seconds
    def init(*args):
        return model.init(*args)

    params = init(rng, state, jnp.ones(input_shape))["params"]  # TODO: why unpack dict? -> for trainstate
    # state = model.reset_state(input_shape)  # we can't use the one we used for init, that is nonzero!
    tx = optax.sgd(config.learning_rate, config.momentum)  # don't have to put on GPU?
    return model, TrainState.create(apply_fn=model.apply, params=params, tx=tx), state


def create_train_state_var(rng, config):
    model = ModelVar()
    input_shape = (config.batch_size, 28, 28, 1)

    @jax.jit  # TODO: this makes a big difference for startup; 5 -> 2 seconds
    def init(*args):
        return model.init(*args)

    state, params = init(rng, jnp.ones(input_shape)).pop("params")
    tx = optax.sgd(config.learning_rate, config.momentum)
    return model, TrainState.create(apply_fn=model.apply, params=params, tx=tx), state


# def create_train_state_stateless(rng, config):
#     model = Stateless()
#     input_shape = (config.batch_size, 28, 28, 1)
#     params = model.init(rng, jnp.ones(input_shape))["params"]
#     tx = optax.sgd(config.learning_rate, config.momentum)
#     return model, TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@functools.partial(jax.jit, static_argnums=(0,))
def train_step(model, state, train_state, data, target):
    print("I'm compiling, and this should only print once!")

    def loss_fn(params, state):
        for _ in range(5):  # TODO: parameterize
            state, logits = model.apply({"params": params}, state, data)
        loss = cross_entropy_loss(logits=logits, labels=target)
        # loss = cross_entropy_loss_vmap(logits=logits, labels=target).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(train_state.params, state)
    train_state = train_state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=target)
    return train_state, metrics


@functools.partial(jax.jit, static_argnums=(0,))
def train_step_var_bw(model, state, train_state, data, target):
    print("I'm compiling, and this should only print once!")

    def loss_fn(params, state):
        for _ in range(5):  # TODO: parameterize
            logits, state = model.apply({"params": params, **state}, data, mutable=["state"])
        loss = cross_entropy_loss(logits=logits, labels=target)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(train_state.params, state)
    train_state = train_state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=target)
    return train_state, metrics


# TODO: gives memory error because of forward-mode AD, so make this runnable!
@functools.partial(jax.jit, static_argnums=(0,))
def train_step_var_fw(model, state, train_state, data, target):
    print("I'm compiling, and this should only print once!")

    def forward_fn(params, state):
        for _ in range(5):  # TODO: parameterize
            logits, state = model.apply({"params": params, **state}, data, mutable=["state"])
        return logits

    def loss_fn(params, state):
        for _ in range(5):  # TODO: parameterize
            logits, state = model.apply({"params": params, **state}, data, mutable=["state"])
        loss = cross_entropy_loss(logits=logits, labels=target)
        return loss

    logits = forward_fn(train_state.params, state)
    grad_fn = jax.jacfwd(loss_fn)
    grads = grad_fn(train_state.params, state)
    train_state = train_state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=target)
    return train_state, metrics


# @functools.partial(jax.jit, static_argnums=(0,))
# def train_step_stateless(model, train_state, data, target):

#     def loss_fn(params):
#         for _ in range(5):  # TODO: parameterize
#             logits = model.apply({"params": params}, data)  # TODO: useless, but for equality
#         loss = cross_entropy_loss(logits=logits, labels=target)
#         return loss, logits

#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (_, logits), grads = grad_fn(train_state.params)
#     train_state = train_state.apply_gradients(grads=grads)
#     metrics = compute_metrics(logits=logits, labels=target)
#     return train_state, metrics


def train_epoch(model, state, train_state, train_loader):
    batch_metrics = []
    for data, target in train_loader:
        # data, target = jax.device_put(data), jax.device_put(target)
        train_state, metrics = train_step(model, state, train_state, data, target)
        batch_metrics.append(metrics)
        # pass

    # compute mean of metrics across each batch in epoch
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    # epoch_metrics_np = {"loss": 0, "accuracy": 0}

    return train_state, epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


def train_epoch_var(model, state, train_state, train_loader):
    batch_metrics = []
    for data, target in train_loader:
        train_state, metrics = train_step_var_bw(model, state, train_state, data, target)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

    return train_state, epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


# def train_epoch_stateless(model, train_state, train_loader):
#     batch_metrics = []
#     for data, target in train_loader:
#         train_state, metrics = train_step_stateless(model, train_state, data, target)
#         batch_metrics.append(metrics)

#     # compute mean of metrics across each batch in epoch
#     batch_metrics_np = jax.device_get(batch_metrics)
#     epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

#     return train_state, epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


@functools.partial(jax.jit, static_argnums=(0,))
def eval_step(model, state, params, data, target):
    for _ in range(5):  # TODO: parameterize
        state, logits = model.apply({"params": params}, state, data)
    return compute_metrics(logits=logits, labels=target)


@functools.partial(jax.jit, static_argnums=(0,))
def eval_step_var(model, state, params, data, target):
    for _ in range(5):  # TODO: parameterize
        logits, state = model.apply({"params": params, **state}, data, mutable=["state"])
    return compute_metrics(logits=logits, labels=target)


# @functools.partial(jax.jit, static_argnums=(0,))
# def eval_step_stateless(model, params, data, target):
#     for _ in range(5):  # TODO: useless, but for equality
#         logits = model.apply({"params": params}, data)
#     return compute_metrics(logits=logits, labels=target)


def eval_model(model, state, params, test_loader):
    batch_metrics = []
    for data, target in test_loader:
        # data, target = jax.device_put(data), jax.device_put(target)
        metrics = eval_step(model, state, params, data, target)
        batch_metrics.append(metrics)
        # pass

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    # epoch_metrics_np = {"loss": 0, "accuracy": 0}

    # why use this below?
    # metrics = jax.device_get(metrics)
    # summary = jax.tree_map(lambda x: x.item(), metrics)

    return epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


def eval_model_var(model, state, params, test_loader):
    batch_metrics = []
    for data, target in test_loader:
        metrics = eval_step_var(model, state, params, data, target)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

    # why use this below?
    # metrics = jax.device_get(metrics)
    # summary = jax.tree_map(lambda x: x.item(), metrics)

    return epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


# def eval_model_stateless(model, params, test_loader):
#     batch_metrics = []
#     for data, target in test_loader:
#         metrics = eval_step_stateless(model, params, data, target)
#         batch_metrics.append(metrics)

#     batch_metrics_np = jax.device_get(batch_metrics)
#     epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

#     # why use this below?
#     # metrics = jax.device_get(metrics)
#     # summary = jax.tree_map(lambda x: x.item(), metrics)

#     return epoch_metrics_np["loss"], epoch_metrics_np["accuracy"]


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
    start_time = time.time()
    # needed for num_workers > 0: https://github.com/google/jax/issues/3382
    # not when using numpy loader
    multiprocessing.set_start_method("spawn")

    # full determinism (https://github.com/google/jax/issues/4823):
    # - provide environment variables: XLA_FLAGS='--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions' TF_CUDNN_DETERMINISTIC=1
    # - for the autotune, either level 1 or 2 should work, not 0, even though that is said to do no autotuning (https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html)
    # - TF_CUDNN_DETERMINISTIC is necessary
    rng = jax.random.PRNGKey(0)
    torch.manual_seed(
        0
    )  # works with mp correctly -> https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

    train_ds = MNIST("data", download=True, train=True, transform=to_numpy)
    test_ds = MNIST("data", download=True, train=False, transform=to_numpy)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=np_collate,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )  # drop_last needed for state as var
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=np_collate,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )
    # TODO: persistent workers is much much faster for spawn with 4 workers
    # not pinning memory seems slightly faster

    summary_writer = SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    # TODO: JAX usually doesn't use model like this and pass it around
    # but I guess it's nice when you have multiple?
    model, train_state, state = create_train_state(init_rng, config)
    # model, train_state, state = create_train_state_var(init_rng, config)
    # model, train_state = create_train_state_stateless(init_rng, config)

    startup_time = time.time()
    print(f"startup time: {startup_time - start_time:.2f}s")

    # jax.profiler.start_trace(workdir)
    for epoch in range(1, config.num_epochs + 1):
        loop_time = time.time()

        train_state, train_loss, train_accuracy = train_epoch(model, state, train_state, train_loader)
        # train_state, train_loss, train_accuracy = train_epoch_var(model, state, train_state, train_loader)
        # train_state, train_loss, train_accuracy = train_epoch_stateless(model, train_state, train_loader)

        train_time = time.time()

        test_loss, test_accuracy = eval_model(model, state, train_state.params, test_loader)
        # test_loss, test_accuracy = eval_model_var(model, state, train_state.params, test_loader)
        # test_loss, test_accuracy = eval_model_stateless(model, train_state.params, test_loader)

        eval_time = time.time()

        print(
            f"epoch:{epoch:3}, time (train/eval): {train_time - loop_time:.2f}s / {eval_time - train_time:.2f}s, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy * 100:.2f}, test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy * 100:.2f}"
        )

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    # jax.profiler.stop_trace()
    summary_writer.flush()


if __name__ == "__main__":
    config = ConfigDict(
        dict(
            learning_rate=0.1,
            momentum=0.9,
            batch_size=1024,
            num_epochs=10,
        )
    )

    # TODO: https://github.com/google/jax/issues/8362
    # and https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    print(
        "make sure to run as 'XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true python tests/mnist_jax.py'"
    )
    train_and_evaluate(config, "./logs")

    # jax.profiler.save_device_memory_profile("memory.prof")  # TODO: use with https://github.com/google/pprof
