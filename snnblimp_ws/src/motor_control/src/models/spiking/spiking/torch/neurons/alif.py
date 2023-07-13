import torch

from spiking.torch.neurons.lif import BaseLIF, SoftLIF


class ALIF(BaseLIF):
    """
    Adaptive LIF neuron with:
    - adaptive threshold
    """

    state_size = 4
    neuron_params = ["leak_i", "leak_v", "leak_t", "base_t", "add_t"]
    

    def activation(self, state, input_, input_spikes):
        # unpack state: spikes always last
        i, v, t, s = state

        # get parameters
        leak_i, leak_v, leak_t = self.get_leaks()
        base_t, add_t, thresh = self.get_thresh(t)  # old thresh for reset

        # current update: leak, integrate
        i = self.update(i, input_, leak_i)

        # voltage update: reset, leak, integrate
        v = self.update_reset(v, i, leak_v, s, thresh)
        
        if self.clamp_v == True:
            v = torch.clamp(v, min=0)

        # threshold update: leak, integrate
        # new thresh for spiking
        t = self.update(t, s, leak_t)
        if self.adapt_thres_input_spikes: t = self.update(t,input_spikes,leak_t) #require the l0 and l1 to be of the same size

        if self.adapt_2x2_connect: thresh = base_t + torch.matmul(t, add_t)
        else: thresh = base_t + add_t * t

        # spike
        s = self.spike(v - thresh)

        return torch.stack([i, v, t, s]), s

    @staticmethod
    def get_spikes(state):
        _, _, _, s = state
        return s

    def get_leaks(self):
        return self.leak_i, self.leak_v, self.leak_t

    def get_thresh(self, t):
        if self.adapt_2x2_connect: #t = (1,N) self.add_t = (N,N)
            thresh = self.base_t +torch.matmul(t, self.add_t)
        else:                      #t = (1,N) self.add_t = (N)
            thresh = self.base_t +self.add_t * t
        return self.base_t, self.add_t, thresh
    




class SoftALIF(ALIF, SoftLIF):
    """
    ALIF neuron with:
    - soft reset of membrane potential
    """
    
    pass
