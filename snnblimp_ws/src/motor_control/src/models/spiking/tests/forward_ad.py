from jax import grad, jacfwd, jvp
import jax.numpy as jnp
import torch
import torch.autograd.forward_ad as fwAD
from torch.autograd.functional import jvp as torch_jvp_bw  # still uses backward-mode AD
from torch.autograd.gradcheck import gradcheck


# relevant discussion: https://github.com/pytorch/pytorch/issues/10223


def multiplication(x):
    return 2 * x


def torch_jvp_fw(f, primals, tangents):
    """
    Adapted from https://github.com/facebookresearch/functorch/blob/main/functorch/_src/eager_transforms.py
    """
    with fwAD.dual_level():
        assert len(primals) == 1 and len(tangents) == 1
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        result_duals = f(*duals)
        primals_out, tangents_out = fwAD.unpack_dual(result_duals[0])
        return primals_out, tangents_out


def torch_jacfwd(f):
    """
    Adapted from https://github.com/facebookresearch/functorch/blob/main/functorch/_src/eager_transforms.py
    """

    def wrapper_fn(primal):
        basis = torch.eye(primal.numel(), dtype=primal.dtype, device=primal.device).view(primal.numel(), *primal.shape)

        def push_jvp(basis):
            _, jvp_out = torch_jvp_fw(f, (primal,), (basis,))
            return jvp_out

        result = torch.vmap(push_jvp)(basis)
        result = result.view(*primal.shape, *primal.shape)
        return result

    return wrapper_fn


if __name__ == "__main__":

    input_ = [1.0, 2.0, 3.0]
    input_jax = jnp.array(input_)
    input_torch = torch.as_tensor(input_)

    print(f"JAX grad(): {grad(multiplication)(1.0)}")
    print(f"JAX jacfwd(): {jacfwd(multiplication)(input_jax)}")
    print(f"JAX jvp(): {jvp(multiplication, [1.0], [0.0])}")

    # with fwAD.dual_level():
    #     input_torch_dual = fwAD.make_dual(input_torch, torch.zeros_like(input_torch))
    #     output = multiplication(input_torch_dual)
    #     print(fwAD.unpack_dual(output)[1])

    print(torch_jvp_fw(multiplication, [input_torch], [torch.zeros_like(input_torch)]))
    print(torch_jacfwd(multiplication)(input_torch))
