import torch

from spiking.torch.neurons.base import BaseNeuron
from spiking.torch.utils.quantization import quantize


class BaseLIF(BaseNeuron):
    """
    Base LIF neuron with:
    - stateful synaptic current
    - hard reset of membrane potential
    - optionally learnable parameters; either per-neuron or single
    """

    state_size = 3
    neuron_params = ["leak_i", "leak_v", "thresh"]

    def __init__(self, fixed_params, learnable_params, spike_fn, layer_setting):
        super().__init__(self.state_size, fixed_params, learnable_params)

        # check parameters are there
        for p in self.neuron_params:
            assert hasattr(self, p), f"{p} not found in {self}"

        self.share_leak_i = layer_setting["shared_leak_i"]
        self.clamp_v = layer_setting["clamp_v"]

        # Only relevant when adaptive lifs are used (then it caclulates t in a different way)
        try: self.adapt_thres_input_spikes = layer_setting["adapt_thres_input_spikes"]
        except: self.adapt_thres_input_spikes = False

        # Only relevant when adaptive lifs are used (then it caclulates t in a different way)
        try: self.adapt_2x2_connect = layer_setting["adapt_2x2_connection"]
        except: self.adapt_2x2_connect = False

        # # spike mechanism
        # self.spike = spike_fn


    def activation(self, state, input_, _):
        # unpack state; spikes always last
        i, v, s = state

        # get parameters
        # TODO: replace with pre-forward hook?
        leak_i, leak_v = self.get_leaks()
        thresh = self.get_thresh()

        # current update: leak, integrate
        i = self.update(i, input_, leak_i)

        # voltage update: leak, reset, integrate
        v = self.update_reset(v, i, leak_v, s, thresh)
        
        if self.clamp_v == True:
            v = torch.clamp(v, min=0)

        # spike
        s = self.spike(v - thresh)

        return torch.stack([i, v, s]), s

    @staticmethod
    def get_spikes(state):
        _, _, s = state
        return s

    def get_leaks(self):
        return self.leak_i, self.leak_v

    def get_thresh(self):
        return self.thresh
    
    @staticmethod
    def spike(v_min_thres):
        return v_min_thres.gt(0).float()


class SoftLIF(BaseLIF):
    """
    LIF neuron with:
    - soft reset of membrane potential
    """

    @staticmethod
    def update_reset(state, input_, leak, reset, thresh):
        return state * leak + input_ - reset * thresh


class LoihiLIF1(BaseLIF):
    """
    LIF neuron that mimics those on Loihi.

    Loihi has:
    - quantized weights in [-256..2..254]
    - quantized leaks in [0..4096] (comparmentCurrentDecay, compartmentVoltageDecay)
    - quantized threshold in [0..131071] (vthMant)
    - factor of 2**6 in current and in threshold
    - voltage reset with new spike afterwards
    - a delay of one timestep between layers, but not at either end of the network

    Either provide fixed quantized parameters, or properly
    initialized learnable parameters + quantization hooks outside.

    Assumes:
    - no bias current (biasMant and biasExp in Loihi set to 0)
    - weightExponent set to 0
    """

    def activation(self, state, input_):
        # unpack state; spikes always last
        i, v, s = state

        # get parameters
        leak_i, leak_v = self.get_leaks()
        thresh = self.get_thresh()

        # current update: leak, integrate
        i = self.update(i, input_, leak_i)
        # voltage update: leak, integrate
        v = self.update(v, i, leak_v)

        # spike
        s = self.spike(v - thresh)

        # voltage reset with new spike
        v = self.reset_v(v, s)

        return torch.stack([i, v, s]), s

    def get_leaks(self):
        leak_i = (4096 - self.leak_i) / 4096
        leak_v = (4096 - self.leak_v) / 4096
        return leak_i, leak_v

    def get_thresh(self):
        return self.thresh * 2 ** 6

    @staticmethod
    def update(state, input_, leak):
        return state * leak + 2 ** 6 * input_

    @staticmethod
    def reset_v(state, reset):
        return state * (1 - reset)


class LoihiLIF2(LoihiLIF1):
    """
    Loihi-compatible version of BaseLIF.

    Because we have spiking neurons with a threshold, we don't actually need to
    scale the parameters to these large ranges. This is nice, because then we don't
    have to scale things like learning rates, surrogate gradients, etc. So:
    - quantized weights in [-1, 1 - 2/256] in steps of 2/256
    - quantized leaks in [0, 1] in steps of 1/4096
    - quantized threshold scaled by same factor as weights, i.e. *1/256 (threshold of 1 should be 256)
    - left out factor of 2**6 in current and in threshold
    - voltage reset with new spike afterwards, but can't do detach in this case (blocks all gradients)
    - a delay buffer that is implemented between layers

    Either provide fixed quantized parameters, or properly
    initialized learnable parameters + quantization hooks outside.

    Assumes:
    - no bias current (biasMant and biasExp in Loihi set to 0)
    - weightExponent set to 0
    """

    def get_leaks(self):
        # no need to scale by 4096
        leak_i = 1 - self.leak_i
        leak_v = 1 - self.leak_v
        return leak_i, leak_v

    def get_thresh(self):
        # no need for 2**6
        return self.thresh

    def update(self, state, input_, leak):
        # no need to multiply input_ by 2**6
        return state * leak + input_
