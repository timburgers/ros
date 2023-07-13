from typing import Any, Dict, Tuple, Type

import flax.linen as nn

from spiking.jax.utils.typing import Array, InitFn


class Layer(nn.Module):
    """
    Layer base class.

    TODO:
    - implement general methods that infer shape for linear and conv layers
    """

    @nn.compact
    def __call__(self, state: Array, input_: Array) -> Array:
        raise NotImplementedError

    def reset_state(self) -> Array:
        raise NotImplementedError
