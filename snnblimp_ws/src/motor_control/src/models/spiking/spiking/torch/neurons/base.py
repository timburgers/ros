import torch
import torch.nn as nn


class BaseNeuron(nn.Module):
    """
    Neuron base class.

    TODO:
    - add base init function for unspecified params inside reset_parameters
    """

    def __init__(self, state_size, fixed_params, learnable_params):
        super().__init__()

        self.state_size = state_size
        self.reset_parameters(fixed_params, learnable_params)

    def forward(self, state, input_, _):
        state = state if state is not None else self.reset_state(input_)
        return self.activation(state, input_, _)

    def activation(self, state, input_, _):
        raise NotImplementedError

    @staticmethod
    def update(state, input_, leak):
        return state * leak + input_

    @staticmethod
    def update_reset(state, input_, leak, reset, thresh):
        return state * leak * (1 - reset) + input_

    def reset_state(self, input_):
        return torch.zeros(self.state_size, *input_.shape, dtype=input_.dtype, device=input_.device)

    def reset_parameters(self, fixed_params, learnable_params):
        # check if disjoint
        assert fixed_params.keys().isdisjoint(learnable_params.keys()), "A parameter cannot be both fixed and learnable"

        # fixed/non-learnable
        for name, data in fixed_params.items():
            # TODO: what happens if already exists?
            self.register_buffer(name, data)

        # learnable
        for name, data in learnable_params.items():
            # TODO: what happens if already exists?
            setattr(self, name, nn.Parameter(data))
