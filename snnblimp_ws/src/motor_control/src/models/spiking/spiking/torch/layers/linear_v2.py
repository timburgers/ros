import torch
import torch.nn as nn

from spiking.torch.neurons.lif import BaseLIF, LoihiLIF1
from spiking.torch.utils.quantization import quantize


class BaseLinear(nn.Module):
    """
    Base densely-connected feedforward linear layer with:
    - no bias
    """

    def __init__(self, input_size, output_size):
        nn.Module.__init__(self)  # needed for multiple inheritance
        self.ff = nn.Linear(input_size, output_size, bias=False)

    def forward(self, state, input_):
        ff = self.ff(input_)
        state = state if state is not None else self.reset_state(ff)

        state, output = self.activation(state, ff)

        return state, output


class BaseRecurrentLinear(nn.Module):
    """
    Base densely-connected recurrent linear layer with:
    - no bias
    """

    def __init__(self, input_size, output_size):
        nn.Module.__init__(self)  # needed for multiple inheritance
        self.ff = nn.Linear(input_size, output_size, bias=False)
        self.rec = nn.Linear(output_size, output_size, bias=False)

    def forward(self, state, input_):
        ff = self.ff(input_)
        state = state if state is not None else self.reset_state(ff)

        s = self.neuron.get_spikes(state)
        rec = self.rec(s)

        state, output = self.activation(state, ff + rec)

        return state, output


class LinearLIF(BaseLinear, BaseLIF):
    """
    Linear layer with LIF activation.
    """

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        BaseLinear.__init__(self, input_size, output_size)
        BaseLIF.__init__(self, fixed_params, learnable_params, spike_fn)


class RecurrentLinearLIF(BaseRecurrentLinear, BaseLIF):
    """
    Recurrent linear layer with LIF activation.
    """

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        BaseLinear.__init__(self, input_size, output_size)
        BaseLIF.__init__(self, fixed_params, learnable_params, spike_fn)


class LinearLoihiLIF(BaseLinear, LoihiLIF1):
    """
    Densely-connected feedforward linear layer of Loihi neurons with:
    - quantized weights in [-256..2..254]
    - TODO: optional axonal delay of 1; Loihi has this between layers, but not at either end
    """

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        BaseLinear.__init__(self, input_size, output_size)
        LoihiLIF1.__init__(self, fixed_params, learnable_params, spike_fn)

        # scale weights
        # TODO: we should be able to parameterize this; provide weights ourselves?
        # or do F.linear ourselves like Norse?
        with torch.no_grad():
            weights = torch.randint_like(self.ff.weight, -256, 254)
            self.ff.weight.copy_(weights)

        # add quantization hook for weights
        quantize(self.ff, "weight", -256, 254, 2)
