import torch.nn as nn

from spiking.torch.neurons.lif import BaseLIF


class LinearLIF(BaseLIF):
    """
    Densely-connected feedforward linear layer with:
    - LIF neuron
    - no bias
    """

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        super().__init__(fixed_params, learnable_params, spike_fn)
        self.ff = nn.Linear(input_size, output_size, bias=False)

    def forward(self, state, input_):
        ff = self.ff(input_)
        state = state if state is not None else self.reset_state(ff)

        state, output = self.activation(state, ff)

        return state, output


class RecurrentLinearLIF(BaseLIF):
    """
    Densely-connected recurrent linear layer with:
    - LIF neuron
    - no bias
    """

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        super().__init__(fixed_params, learnable_params, spike_fn)
        self.ff = nn.Linear(input_size, output_size, bias=False)
        self.rec = nn.Linear(output_size, output_size, bias=False)

    def forward(self, state, input_):
        ff = self.ff(input_)
        state = state if state is not None else self.reset_state(ff)

        s = self.get_spikes(state)
        rec = self.rec(s)

        state, output = self.activation(state, ff + rec)

        return state, output
