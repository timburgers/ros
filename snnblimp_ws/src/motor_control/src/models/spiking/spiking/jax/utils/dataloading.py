"""
Utils for use with PyTorch data handling (DataLoader and Dataset).
"""

import jax
import jax.numpy as jnp
import numpy as np


# TODO: these JAX variants are much much slower, why?
# def jax_transform(input_):
#     return jnp.expand_dims(jnp.asarray(input_, dtype=jnp.float32) / 255, -1)  # flax uses nhwc instead of nchw (torch)


# def jax_collate(batch):
#     if isinstance(batch[0], jnp.ndarray):
#         return jnp.stack(batch)
#     elif isinstance(batch[0], (tuple, list)):
#         return type(batch[0])(jax_collate(samples) for samples in zip(*batch))
#     else:
#         return jnp.asarray(batch)


# def to_numpy(input_):
#     """
#     Transform to convert to NumPy array.
#     Intended for use after torchvision.transforms.ToTensor.
#     """
#     dtype = jax.dtypes.canonicalize_dtype(jnp.float_)
#     return np.asarray(input_, dtype=dtype).swapaxes(0, -1)  # chw -> hwc


def to_numpy(pic):
    """
    Transform to convert to NumPy array.
    Analogous to torchvision.transforms.ToTensor.
    """
    dtype = jax.dtypes.canonicalize_dtype(jnp.float_)

    # handle numpy array
    if isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            img = pic[:, :, None]  # hwc

        # backward compatibility
        if img.dtype == np.byte:
            return img.astype(dtype) / 255
        else:
            return img.astype(dtype)

    # handle PIL Image
    mode_to_nptype = {"I": jnp.int32, "I;16": jnp.int16, "F": jnp.float32}
    # ToTensor has copy here, seems a bit faster, so let's copy instead of np.asarray
    img = np.array(pic, dtype=mode_to_nptype.get(pic.mode, jnp.uint8), copy=True)

    if pic.mode == "1":
        img = 255 * img
    img = img.reshape((pic.size[1], pic.size[0], len(pic.getbands())))  # hwc
    if img.dtype == np.byte:
        return img.astype(dtype) / 255
    else:
        return img.astype(dtype)


def np_collate(batch):
    """
    NumPy collate function.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(np_collate(samples) for samples in zip(*batch))
    else:
        return np.array(batch)
