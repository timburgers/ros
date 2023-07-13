import torch

from spiking.torch.utils.distributions import gaussian


# TODO: benchmark this vs a specific function per surrogate: is eval() making it slower?
def get_spike_fn(name, *args):
    args = [torch.tensor(arg) for arg in args]

    def inner(x):
        #Replace eval(name) for BaseSpike due to error when using evotorch lib
        return BaseSpike.apply(x, *args)
    return inner


class BaseSpike(torch.autograd.Function):
    """
    Base spike function without gradient.
    """

    @staticmethod
    def forward(ctx, x, *args):
        ctx.save_for_backward(x, *args)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class STE(torch.autograd.Function):
    """
    Spike function with straight-through estimator as gradient.

    Originally proposed by Hinton in his 2012 Coursera lectures.
    """

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class SaturatedSTE(torch.autograd.Function):
    """
    Spike function with saturated straight-through estimator as gradient (boxcar).

    From "Binarized Neural Networks", Hubara and Courbariaux et al., NIPS 2016.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = x.abs().le(bound).float()
        return grad_input * mask, None


class LowerBoundSTE(torch.autograd.Function):
    """
    Spike function with straight-through-estimated gradient that doesn't
    propagate when inputs are negative (derivative of ReLU).

    From "Training Deep Spiking Auto-encoders...", Huebotter et al., arXiv 2021.
    Equivalent to SpikeOperator surrogate in snnTorch: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors  # no kwargs in forward()
        grad_input = grad_output.clone()
        mask = x.gt(0).float()
        return grad_input * mask


class ProductSTE(torch.autograd.Function):
    """
    Spike function with gradient that is the product of input and incoming gradient.

    From snnTorch repository: https://github.com/jeshraghian/snntorch/blob/master/snntorch/__init__.py.
    """

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors  # no kwargs in forward()
        grad_input = grad_output.clone()
        return grad_input * x, None


class FastSigmoid(BaseSpike):
    """
    Spike function with derivative of fast sigmoid as surrogate gradient (SuperSpike)

    From "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al., Neural Computation 2018.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = height / (1 + slope * x.abs()) ** 2
        return grad_input * sg, None, None


class Sigmoid(BaseSpike):
    """
    Spike function with derivate of sigmoid as surrogate gradient.

    From TODO
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        sigmoid = height / (1 + torch.exp(-slope * x))
        sg = sigmoid * (1 - sigmoid)
        return grad_input * sg, None, None


class SparseFastSigmoid(BaseSpike):
    """
    Spike function with derivative of fast sigmoid as surrogate gradient,
    but clamped around the threshold to ensure sparse gradients.

    From "Sparse Spiking Gradient Descent", Perez-Nieves et al., arXiv 2021.

    TODO: CUDA implementation that saves a lot (as in paper)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, bound, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = x.abs().lt(bound).float()
        sg = height / (1 + slope * x.abs()) ** 2
        return grad_input * sg * mask, None, None, None


class MultiGaussian(BaseSpike):
    """
    Spike function with combination-of-Gaussians surrogate gradient.

    From "Accurate and efficient time-domain classification...", Yin et al. NMI 2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
        sg = (
            1.15 * gaussian(x, zero, width)
            - 0.15 * gaussian(x, width, 6 * width)
            - 0.15 * gaussian(x, -width, 6 * width)
        )
        return grad_input * sg, None


class Triangle(BaseSpike):
    """
    Spike function with triangulur/piecewise-linear surrogate gradient.

    From "Convolutional networks for fast, energy-efficient neuromorphic computing", Esser et al., PNAS 2016.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(height - slope * x.abs())
        return grad_input * sg, None, None


class ArcTan(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.

    From "Incorporating Learnable Membrane Time Constant...", Fang et al., arXiv 2020.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, height, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = height / (1 + slope * x * x)
        return grad_input * sg, None, None


class LeakyReLU(BaseSpike):
    """
    Spike function with derivative of leaky ReLU as surrogate gradient.

    From snnTorch repository: https://github.com/jeshraghian/snntorch/blob/master/snntorch/surrogate.py
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, slope = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = x.gt(0).float()
        grad = grad_input * mask + (1 - mask) * slope * grad_input
        return grad, None
