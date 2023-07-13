import torch.nn as nn

from spiking.torch.neurons.lif import BaseLIF


class BaseConv(nn.Module):
    """
    Base convolutional feedforward layer with:
    - no bias

    TODO:
    - can we merge this with BaseLinear into a BaseLayer, providing layer-specific args as a dict?
    """

    neuron_model = None

    def __init__(self, in_channels, out_channels, kernel_size, stride, fixed_params, learnable_params, spike_fn):
        super().__init__()

        self.ff = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.neuron = self.neuron_model(fixed_params, learnable_params, spike_fn)

    def forward(self, state, input_):
        ff = self.ff(input_)
        state, output = self.neuron(state, ff)

        return state, output


class ConvLIF(BaseConv):
    """
    Convolutional layer with LIF activation.
    """

    neuron_model = BaseLIF
