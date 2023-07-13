from typing import Tuple

import flax.linen as nn

from spiking.jax.utils.typing import Array, Shape


class Neuron(nn.Module):
    """
    Neuron base class.
    """

    size: int

    @nn.compact
    def __call__(self, state: Array, input_: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    @staticmethod
    def reset_state(shape: Shape) -> Array:
        raise NotImplementedError

    @staticmethod
    def get_output(state: Array) -> Array:
        raise NotImplementedError

    @staticmethod
    def update(state: Array, input_: Array, leak: Array) -> Array:
        return state * leak + input_

    @staticmethod
    def update_reset(state: Array, input_: Array, leak: Array, reset: Array, thresh: Array) -> Array:
        return state * leak * (1 - reset) + input_
