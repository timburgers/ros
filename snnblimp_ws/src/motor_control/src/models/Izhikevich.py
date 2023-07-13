import torch

from spiking.torch.neurons.base import BaseNeuron
from spiking.torch.layers.linear import BaseLinear



class BaseIzhikevich(BaseNeuron):
    """
    Base Izhickevich neuron with:
    - stateful mem potential and recovery variable
    - hard reset of membrane potential
    - optionally learnable parameters; either per-neuron or single
    """

    # recovery variable (u), membrane potential (v) output spike (s)
    state_size = 3
    neuron_params = ["a","b","c","d" "thresh","v2", "v1", "v0", "tau_u"]
    

    def __init__(self, fixed_params, learnable_params, spike_fn, *dt):
        super().__init__(self.state_size, fixed_params, learnable_params)

        # check parameters are there
        for p in ["a","b","c","d", "thresh","v2", "v1", "v0", "tau_u"]:
            assert hasattr(self, p), f"{p} not found in {self}"

        self.dt =dt[0]
        # spike mechanism for back prop
        self.spike = spike_fn

    def activation(self, state, input_):
        # unpack state; spikes always last
        u, v, s = state

        # get parameters
        # TODO: replace with pre-forward hook?
        a,b,c,d,v2,v1,v0,tau_u = self.get_param()
        thresh = self.get_thresh()


        # voltage update + reset + integrate
        v = self.update_mem(v, u, input_, s, c, self.dt, v2, v1, v0, tau_u)

        # recovery update + reset
        u = self.update_recov(a, b, d, v, u, s, self.dt)


        # spike
        s = self.spike(v - thresh)

        return torch.stack([u, v, s]), s

    @staticmethod
    def get_spikes(state):
        _, _, s = state
        return s

    def get_param(self):
        return self.a, self.b, self.c, self.d, self.v2, self.v1, self.v0, self.tau_u

    def get_thresh(self):
        return self.thresh
    
    def get_time_step(self):
        return self.time_step

    @staticmethod
    def update_recov(a,b,d,v,u,reset,dt):
        # The parameters, thres are trained 
        return u + (dt)*a*(b*v - u) + d*reset
    
    @staticmethod
    def update_mem(v,u,I,reset,c,dt, v2, v1, v0, tau_u):
        # The parameters, thres are trained. dt is multiplied with 1000 since all parameters are defined in ms
        return (v + (dt)*(v2*v**2 + v1*v + v0 -tau_u*u + I))*(1-reset) + reset*c



class LinearIzhikevich(BaseLinear):
    """
    Linear layer with Izhikevich activation.
    """

    neuron_model = BaseIzhikevich