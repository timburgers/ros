import torch
import torch.nn as nn

from spiking.torch.neurons.lif import BaseLIF, LoihiLIF2, SoftLIF
from spiking.torch.neurons.alif import ALIF, SoftALIF
from spiking.torch.utils.quantization import quantize


class BaseLinear(nn.Module):
    """
    Base densely-connected feedforward linear layer with:
    - no bias
    """

    neuron_model = None

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn, layer_setting, layer_name):
        super().__init__()
        #Check which layer this is, since weights are structured differently
        self.layer_name = layer_name
        self.input_size = input_size
        self.output_size = output_size
            
        try: self.w_diagonal = layer_setting["w_diagonal"]                         
        except: self.w_diagonal = False

        try: self.w_diagonal_2x2 = layer_setting["w_diagonal_2x2"]
        except: self.w_diagonal_2x2 = False

        try: self.w2x2_shared_cross = layer_setting["shared_2x2_weight_cross"]
        except: self.w2x2_shared_cross = False

        try: self.shared_leak_iv = layer_setting["shared_leak_iv"]            
        except: self.shared_leak_iv = False

        try: self.shared_thres = layer_setting["shared_thres"]
        except: self.shared_thres = False

        try: self.adaptive = layer_setting["adaptive"]                      
        except: self.adaptive = False

        try: self.adapt_share_baseleak_t = layer_setting["adapt_share_baseleak_t"]
        except: self.adapt_share_baseleak_t = False

        try: self.adapt_2x2_connect = layer_setting["adapt_2x2_connection"]
        except: self.adapt_2x2_connect = False

        try: self.share_add_t = layer_setting["adapt_share_add_t"]
        except: self.share_add_t = False



        # Set the bias for layer 1 from config and for layer 2 fto false
        self.bias_enabled = layer_setting["bias"]
        self.weight_and_bias_shared = layer_setting["shared_weight_and_bias"]

        self.ff = nn.Linear(input_size, output_size, bias=self.bias_enabled) #layer itself is not used
        self.neuron = self.neuron_model(fixed_params, learnable_params, spike_fn,layer_setting)


    # Call this function, after the weight and biases are initialize
    def init_reshape(self):

        ### RESHAPE WEIGHT MATRIX
        if self.layer_name == "start" or self.layer_name =="end":
            # First/Last layer, Shared W&B
            if self.weight_and_bias_shared:
                self.weight = torch.flatten(torch.stack((self.ff.weight,-1*self.ff.weight),dim=1)).reshape(self.output_size, self.input_size) 
            
            # First/Last layer, NO Shared W&B
            else: 
                self.weight = self.ff.weight.reshape(self.output_size, self.input_size) 

        # Middle layer
        else:
            # Middle layer, Diagonal
            if self.w_diagonal:
                # Middle layer, Diagonal (2x2)
                if self.w_diagonal_2x2:
                    # Middle layer, Diagonal (2x2), Shared W&B
                    if self.w2x2_shared_cross:
                        ind_param = 0
                        self.weight = None
                        for idx_block in range(int(self.output_size/2)):
                            block_2x2 = torch.tensor([[self.ff.weight[ind_param], self.ff.weight[ind_param+1]],[self.ff.weight[ind_param+1], self.ff.weight[ind_param]]]) #shape [[x0, x1],[x1,x0]]
                            self.weight = block_2x2 if self.weight == None else torch.block_diag(self.weight, block_2x2)
                            ind_param += 2

                    elif self.weight_and_bias_shared:
                        ind_param = 0
                        self.weight = None
                        for idx_block in range(int(self.output_size/2)):
                            block_2x2 = torch.tensor([[1, 1],[1, 1]])*self.ff.weight[ind_param]
                            self.weight = block_2x2 if self.weight == None else torch.block_diag(self.weight, block_2x2)
                            ind_param += 1

                    # Middle layer, Diagonal (2x2), Shared W&B
                    else:
                        ind_param_start = 0
                        self.weight = None
                        for idx_block in range(int(self.output_size/2)):
                            block_2x2 = self.ff.weight[ind_param_start:ind_param_start+4].reshape((2,2))
                            self.weight = block_2x2 if self.weight == None else torch.block_diag(self.weight, block_2x2)
                            ind_param_start += 4

                # Middle layer, Diagonal (1x1)
                else:
                    # Middle layer, Diagonal (1x1), Shared W&B
                    if self.weight_and_bias_shared:
                        self.weight = torch.flatten(torch.stack((self.ff.weight,self.ff.weight),dim=1)) 
                        self.weight = torch.diag(self.weight) 
                        

                    # Middle layer, Diagonal (1x1), NON Shared W&B
                    else:
                        self.weight = torch.diag(self.ff.weight) 
                
            # Middle layer, Non diagonal
            else:
                #Middle layer, Non diagonal, Shared W&B
                if self.weight_and_bias_shared:
                    self.weight = torch.flatten(torch.stack((self.ff.weight,self.ff.weight),dim=1)).reshape(int(self.output_size/2), self.input_size)
                    self.weight = torch.flatten(torch.stack((self.weight,self.weight),dim=1)).reshape(self.output_size, self.input_size)      
                
                #Middle layer, Non diagonal, Non Shared W&B
                else:
                    self.weight = self.ff.weight.reshape(self.output_size, self.input_size) 

        ### RESHAPE BIAS MATRIX
        if self.bias_enabled:
            if self.weight_and_bias_shared:
                self.bias = torch.flatten(torch.stack((self.ff.bias,self.ff.bias),dim=1))
            else:
                self.bias = torch.tensor(self.ff.bias)
        
        # results in the is_leaf_True, which is required for the deep copy
        self.ff.weight = torch.nn.Parameter(self.weight) 
        self.ff.bias = torch.nn.Parameter(self.bias) if self.bias_enabled else None

        ### RESHAPE LEAK_I MATRIX
        if self.shared_leak_iv:
            self.neuron.leak_i = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.leak_i,self.neuron.leak_i),dim=1)))
            self.neuron.leak_v = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.leak_v,self.neuron.leak_v),dim=1)))
        
        ### RESHAPE THRESHOLD
        if self.shared_thres:
            self.neuron.thresh = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.thresh,self.neuron.thresh),dim=1)))
        
        ### RESHAPE ADD_T MATRIX
        if self.adaptive:
            if self.adapt_2x2_connect:
                if self.share_add_t:
                    ind_param = 0
                    add_t_ = None
                    for idx_block in range(int(self.output_size/2)):
                        block_2x2 = torch.tensor([[-1,1],[1,-1]])*self.neuron.add_t[ind_param]
                        add_t_ = block_2x2 if add_t_ == None else torch.block_diag(add_t_, block_2x2)
                        ind_param += 1
                    self.neuron.add_t = torch.nn.Parameter(add_t_)

                else:
                    ind_param_start = 0
                    add_t_ = None
                    for idx_block in range(int(self.output_size/2)):
                        block_2x2 = self.neuron.add_t[ind_param_start:ind_param_start+4].reshape((2,2))
                        add_t_ = block_2x2 if add_t_ == None else torch.block_diag(add_t_, block_2x2)
                        ind_param_start += 4
                    self.neuron.add_t = torch.nn.Parameter(add_t_)
        
        ### RESHAPE BASE AND LEAK
        if self.adaptive and self.adapt_share_baseleak_t:
            self.neuron.base_t = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.base_t,self.neuron.base_t),dim=1)))
            self.neuron.leak_t = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.leak_t,self.neuron.leak_t),dim=1)))



    def forward(self, state, input_):
        # First/Last layer

        ff =torch.nn.functional.linear(input_,self.ff.weight,self.ff.bias)
        
        state, output = self.neuron(state, ff, input_)

        return state, output








class BaseRecurrentLinear(nn.Module):
    """
    Base densely-connected recurrent linear layer with:
    - no bias
    """

    neuron_model = None

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn, layer_setting,layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.input_size = input_size
        self.output_size = output_size

        try: self.w_diagonal = layer_setting["w_diagonal"]
        except: self.w_diagonal = False

        try: self.rec_2x2 = layer_setting["recurrent_2x2"]
        except: self.rec_2x2 = False

        try: self.w_diagonal_2x2 = layer_setting["w_diagonal_2x2"]
        except: self.w_diagonal_2x2 = False

        try: self.shared_leak_iv = layer_setting["shared_leak_iv"]
        except: self.shared_leak_iv = False

        try: self.shared_thres = layer_setting["shared_thres"]
        except: self.shared_thres = False

        try: self.w2x2_shared_cross = layer_setting["shared_2x2_weight_cross"]
        except: self.w2x2_shared_cross = False

        try: self.adaptive = layer_setting["adaptive"]                      
        except: self.adaptive = False

        try: self.adapt_2x2_connect = layer_setting["adapt_2x2_connection"]
        except: self.adapt_2x2_connect = False

        try: self.share_add_t = layer_setting["adapt_share_add_t"]
        except: self.share_add_t = False

        try: self.adapt_share_baseleak_t = layer_setting["adapt_share_baseleak_t"]
        except: self.adapt_share_baseleak_t = False

        self.bias_enabled = layer_setting["bias"]
        self.weight_and_bias_shared = layer_setting["shared_weight_and_bias"]


        
        self.ff = nn.Linear(input_size, output_size, bias=self.bias_enabled)
        self.rec = nn.Linear(output_size, output_size, bias=False)
        self.neuron = self.neuron_model(fixed_params, learnable_params, spike_fn, layer_setting)
    
    # Call this function, after the weight and biases are initialize
    def init_reshape(self):

        ### RESHAPE WEIGHT MATRIX
        if self.layer_name == "start" or self.layer_name =="end":
            # First/Last layer, Shared W&B
            if self.weight_and_bias_shared:
                self.weight = torch.flatten(torch.stack((self.ff.weight,-1*self.ff.weight),dim=1)).reshape(self.output_size, self.input_size) 
            
            # First/Last layer, NO Shared W&B
            else: 
                self.weight = self.ff.weight.reshape(self.output_size, self.input_size) 

        # Middle layer
        else:
            # Middle layer, Diagonal
            if self.w_diagonal:
                # Middle layer, Diagonal (2x2)
                if self.w_diagonal_2x2:
                    # Middle layer, Diagonal (2x2), Shared W&B
                    if self.w2x2_shared_cross:
                        ind_param = 0
                        self.weight = None
                        for idx_block in range(int(self.output_size/2)):
                            block_2x2 = torch.tensor([[self.ff.weight[ind_param], self.ff.weight[ind_param+1]],[self.ff.weight[ind_param+1], self.ff.weight[ind_param]]]) #shape [[x0, x1],[x1,x0]]
                            self.weight = block_2x2 if self.weight == None else torch.block_diag(self.weight, block_2x2)
                            ind_param += 2

                    elif self.weight_and_bias_shared:
                        ind_param = 0
                        self.weight = None
                        for idx_block in range(int(self.output_size/2)):
                            block_2x2 = torch.tensor([[1, 1],[1, 1]])*self.ff.weight[ind_param]
                            self.weight = block_2x2 if self.weight == None else torch.block_diag(self.weight, block_2x2)
                            ind_param += 1

                    # Middle layer, Diagonal (2x2), Shared W&B
                    else:
                        ind_param_start = 0
                        self.weight = None
                        for idx_block in range(int(self.output_size/2)):
                            block_2x2 = self.ff.weight[ind_param_start:ind_param_start+4].reshape((2,2))
                            self.weight = block_2x2 if self.weight == None else torch.block_diag(self.weight, block_2x2)
                            ind_param_start += 4

                # Middle layer, Diagonal (1x1)
                else:
                    # Middle layer, Diagonal (1x1), Shared W&B
                    if self.weight_and_bias_shared:
                        self.weight = torch.flatten(torch.stack((self.ff.weight,self.ff.weight),dim=1)) 
                        self.weight = torch.diag(self.weight) 
                        

                    # Middle layer, Diagonal (1x1), NON Shared W&B
                    else:
                        self.weight = torch.diag(self.ff.weight) 
                
            # Middle layer, Non diagonal
            else:
                #Middle layer, Non diagonal, Shared W&B
                if self.weight_and_bias_shared:
                    self.weight = torch.flatten(torch.stack((self.ff.weight,self.ff.weight),dim=1)).reshape(int(self.output_size/2), self.input_size)
                    self.weight = torch.flatten(torch.stack((self.weight,self.weight),dim=1)).reshape(self.output_size, self.input_size)      
                
                #Middle layer, Non diagonal, Non Shared W&B
                else:
                    self.weight = self.ff.weight.reshape(self.output_size, self.input_size) 

        ### RESHAPE BIAS MATRIX
        if self.bias_enabled:
            if self.weight_and_bias_shared:
                self.bias = torch.flatten(torch.stack((self.ff.bias,self.ff.bias),dim=1))
            else:
                self.bias = torch.tensor(self.ff.bias)
        
        # results in the is_leaf_True, which is required for the deep copy
        self.ff.weight = torch.nn.Parameter(self.weight) 
        self.ff.bias = torch.nn.Parameter(self.bias) if self.bias_enabled else None

        ### RESHAPE LEAK_I MATRIX
        if self.shared_leak_iv:
            self.neuron.leak_i = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.leak_i,self.neuron.leak_i),dim=1)))
            self.neuron.leak_v = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.leak_v,self.neuron.leak_v),dim=1)))

        ### RESHAPE THRESHOLD
        if self.shared_thres:
            self.neuron.thresh = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.thresh,self.neuron.thresh),dim=1)))

        ### RESHAPE ADD_T MATRIX
        if self.adaptive:
            if self.adapt_2x2_connect:
                if self.share_add_t:
                    ind_param = 0
                    add_t_ = None
                    for idx_block in range(int(self.output_size/2)):
                        block_2x2 = torch.tensor([[-1,1],[1,-1]])*self.neuron.add_t[ind_param]
                        add_t_ = block_2x2 if add_t_ == None else torch.block_diag(add_t_, block_2x2)
                        ind_param += 1
                    self.neuron.add_t = torch.nn.Parameter(add_t_)

                else:
                    ind_param_start = 0
                    add_t_ = None
                    for idx_block in range(int(self.output_size/2)):
                        block_2x2 = self.neuron.add_t[ind_param_start:ind_param_start+4].reshape((2,2))
                        add_t_ = block_2x2 if add_t_ == None else torch.block_diag(add_t_, block_2x2)
                        ind_param_start += 4
                    self.neuron.add_t = torch.nn.Parameter(add_t_)
        
        ### RESHAPE BASE AND LEAK
        if self.adaptive and self.adapt_share_baseleak_t:
            self.neuron.base_t = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.base_t,self.neuron.base_t),dim=1)))
            self.neuron.leak_t = torch.nn.Parameter(torch.flatten(torch.stack((self.neuron.leak_t,self.neuron.leak_t),dim=1)))
        
        ### RESHAPE BASE AND LEAK
        if self.rec_2x2:
            ind_param_start = 0
            self.rec_weight = None
            for idx_block in range(int(self.output_size/2)):
                block_2x2 = self.rec.weight[ind_param_start:ind_param_start+4].reshape((2,2))
                self.rec_weight = block_2x2 if self.rec_weight == None else torch.block_diag(self.rec_weight, block_2x2)
                ind_param_start += 4
            self.rec.weight = torch.nn.Parameter(self.rec_weight) 



    def forward(self, state, input_):

        ff =torch.nn.functional.linear(input_,self.ff.weight,self.ff.bias)
        
        state = state if state is not None else self.neuron.reset_state(ff)

        s = self.neuron.get_spikes(state)
        rec = self.rec(s)

        state, output = self.neuron.activation(state, ff + rec, input_)  # .activation to save a state check
        return state, output


class LinearLIF(BaseLinear):
    """
    Linear layer with LIF activation.
    """
    neuron_model = SoftLIF

class LinearALIF(BaseLinear):
    """
    Linear layer with ALIF activation.
    """
    neuron_model = SoftALIF

    

class RecurrentLinearLIF(BaseRecurrentLinear):
    """
    Recurrent linear layer with LIF activation.
    """
    neuron_model = SoftLIF


class RecurrentLinearALIF(BaseRecurrentLinear):
    """
    Recurrent linear layer with LIF activation.
    """

    neuron_model = SoftALIF

class LinearLoihiLIF2(BaseLinear):
    """
    Densely-connected feedforward linear layer of Loihi-compatible neurons with:
    - quantized weights in [-1, 1 - 2/256]
    - TODO: optional axonal delay of 1; Loihi has this between layers, but not at either end
    """

    neuron_model = LoihiLIF2

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        super().__init__(input_size, output_size, fixed_params, learnable_params, spike_fn)

        # add quantization hook for weights
        quantize(self.ff, "weight", -1, 1 - 2 / 256, 2 / 256)
