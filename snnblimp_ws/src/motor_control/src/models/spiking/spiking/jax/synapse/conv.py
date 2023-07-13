from typing import Any, Callable, Iterable, Tuple, Union

import flax.linen as nn
import jax.lax as lax
import jax.numpy as jnp

import spiking.jax.synapse.base as base
from spiking.jax.utils.typing import Array, Dtype, PRNGKey, Shape


class ConvPatches(base.Synapse):
    """
    Convolution Module that returns both the non-overlapping input patches
    as well as the resulting convolution between input and kernel.

    Follows flax.linen.Conv.
    """

    features: int
    kernel_size: Iterable[int]
    strides: Union[None, int, Iterable[int]] = 1
    padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
    input_dilation: Union[None, int, Iterable[int]] = 1
    kernel_dilation: Union[None, int, Iterable[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input_: Array) -> Tuple[Array, Array]:
        input_ = jnp.asarray(input_, self.dtype)

        if isinstance(self.kernel_size, int):
            raise TypeError("kernel_size must be an iterable")
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x):
            if x is None:
                x = 1
            if isinstance(x, int):
                x = (x,) * len(kernel_size)
            return x

        is_single_input = False
        if input_.ndim == len(kernel_size) + 1:
            is_single_input = True
            input_ = jnp.expand_dims(input_, axis=0)

        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        in_features = input_.shape[-1]
        assert in_features % self.feature_group_count == 0
        kernel_shape = kernel_size + (in_features // self.feature_group_count, self.features)
        kernel = self.param("kernel", self.kernel_init, kernel_shape)
        kernel = jnp.asarray(kernel, self.dtype)

        dimension_numbers = nn.linear._conv_dimension_numbers(input_.shape)
        patches = lax.conv_general_dilated_patches(
            input_,
            kernel_size,
            strides,
            self.padding,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            precision=self.precision,
        )
        y = jnp.matmul(patches, kernel.reshape(-1, self.features))

        if is_single_input:
            patches = jnp.squeeze(patches, axis=0)
            y = jnp.squeeze(y, axis=0)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y += bias

        return y, patches
