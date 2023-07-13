import flax.linen as nn

from spiking.jax.utils.typing import Array


class Synapse(nn.Module):
    """
    Synapse base class.
    """

    @nn.compact
    def __call__(self, input_: Array) -> Array:
        raise NotImplementedError
