import torch
import torch.nn as nn
from spiking.torch.layers.linear import RecurrentLinearLIF, LinearLIF, LinearALIF, RecurrentLinearALIF
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from models.Leaky_integrator import Linear_LI_filter
import numpy as np


class L1_Decoding_SNN(nn.Module):
	def __init__(self, param_init, neurons,layer_settings):
		super(L1_Decoding_SNN,self).__init__()

		# Select the number of inputs to the first layer depending wheter encoding layer is present
		if layer_settings["l0"]["enabled"]: 
			self.encoding_layer = True
			self.l1_input = neurons if layer_settings["l1"]["w_diagonal"] else layer_settings["l0"]["neurons"]
			self.l1_name = "middle"
		else:
			self.encoding_layer = False 
			self.l1_input = 1
			self.l1_name = "start"
		
		self.l2_name = "end"
		self.l1_output = neurons
		self.neurons = neurons
		self.params_fixed_l1= dict()
		self.params_fixed_l2 = dict()
		self.param_init = param_init

		# Initialize the param_init array
		if self.param_init == None: self.param_init = init_l1_l2(neurons,layer_settings)


		# Init the parameters that are all present
		self.params_learnable_l1=dict(leak_i = self.param_init["l1_leak_i"], leak_v = self.param_init["l1_leak_v"]) 
		self.params_learnable_l2 = dict(leak = self.param_init["l2_leak"])


		# Different parameters for adaptive/non-adaptive neurons
		if layer_settings["l1"]["adaptive"]: 
			self.params_learnable_l1["leak_t"] = self.param_init["l1_leak_t"]
			self.params_learnable_l1["base_t"] = self.param_init["l1_base_t"]
			self.params_learnable_l1["add_t"] = self.param_init["l1_add_t"]
		else:
			self.params_learnable_l1["thresh"] = self.param_init["l1_thres"]
		

		# Initializing the first layer
		if layer_settings["l1"]["recurrent"]:
			if layer_settings["l1"]["adaptive"]:
				self.l1 = RecurrentLinearALIF(self.l1_input,self.l1_output,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l1"], self.l1_name)
			else:
				self.l1 = RecurrentLinearLIF(self.l1_input,self.l1_output,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l1"], self.l1_name)
			self.l1.rec.weight = torch.nn.parameter.Parameter(self.param_init["l1_weights_rec"])
		else:
			if layer_settings["l1"]["adaptive"]:
				self.l1 = LinearALIF(self.l1_input,self.l1_output,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l1"], self.l1_name)
			else: 
				self.l1 = LinearLIF(self.l1_input,self.l1_output,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l1"], self.l1_name)


		# Initializing the second layer
		layer_settings["l2"]["bias"] = False
		self.l2 = Linear_LI_filter(self.l1_output,1,self.params_fixed_l2,self.params_learnable_l2,None, layer_settings["l2"], self.l2_name)


		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		self.l1.ff.weight = torch.nn.parameter.Parameter(self.param_init["l1_weights"])
		self.l2.ff.weight = torch.nn.parameter.Parameter(self.param_init["l2_weights"])
		if layer_settings["l1"]["bias"] == True: self.l1.ff.bias = torch.nn.parameter.Parameter(self.param_init["l1_bias"])


	def forward(self, input_batch, state_l1, state_l2):

		batch_size, seq_length, n_inputs = input_batch.size()
		# Different pass if the for loop is outside of the forward pass aka, seq_len is one
		if seq_length ==1:
			input = input_batch
			state_l1,spikes_l1 = self.l1(state_l1,input)
			state_l2, _ = self.l2(state_l2,spikes_l1)
		
		# if forward pass is inside this pass, aka seq_len is larger than 1
		else:
			state_l1_list, state_l2_list = [],[]
			for timestep in range(seq_length):
				input = input_batch[:,timestep,:]
				state_l1, spikes_l1 = self.l1(state_l1,input)
				state_l2, _ = self.l2(state_l2,spikes_l1)

				state_l1_list += [state_l1]
				state_l2_list += [state_l2]

			state_l1 = torch.stack(state_l1_list)
			state_l2 = torch.stack(state_l2_list)
		
		return state_l1, state_l2

class Encoding_L1_Decoding_SNN(L1_Decoding_SNN):
	def __init__(self, param_init, neurons,layer_settings):
		super(Encoding_L1_Decoding_SNN,self).__init__(param_init, neurons,layer_settings)
		self.params_fixed_l0= dict()
		self.l0_name = "start"
		
		# Initialize the param_init array
		if param_init == None: param_init = init_l0_l1_l2(neurons,layer_settings,self.param_init)
		self.param_init = param_init

		# Init the parameters that are all present
		self.params_learnable_l0=dict(leak_i = self.param_init["l0_leak_i"], leak_v = self.param_init["l0_leak_v"], thresh = self.param_init["l0_thres"]) 

		self.l0 = LinearLIF(1,self.l1_input,self.params_fixed_l0,self.params_learnable_l0,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l0"], self.l0_name)

		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		self.l0.ff.weight = torch.nn.parameter.Parameter(param_init["l0_weights"])
		if layer_settings["l0"]["bias"]: self.l0.ff.bias = torch.nn.parameter.Parameter(param_init["l0_bias"])
			

	def forward(self, input_, state_l0, state_l1, state_l2):
		input = input_
		state_l0,spikes_l0 = self.l0(state_l0,input)
		state_l1,spikes_l1 = self.l1(state_l1,spikes_l0)
		state_l2, _ 		= self.l2(state_l2,spikes_l1)
		
		return state_l0, state_l1, state_l2

	
def init_l1_l2(neurons,layer_set):
	# Set local variables
	init_param = {}
	l1_adapt				= layer_set["l1"]["adaptive"]
	l1_recur				= layer_set["l1"]["recurrent"]
	l1_recur2x2				= layer_set["l1"]["recurrent_2x2"]
	encoding_layer			= layer_set["l0"]["enabled"]
	l1_shared_wb 			= layer_set["l1"]["shared_weight_and_bias"]
	l2_shared_wb 			= layer_set["l2"]["shared_weight_and_bias"]
	l1_w_diagonal 			= layer_set["l1"]["w_diagonal"]
	l1_w_2x2				= layer_set["l1"]["w_diagonal_2x2"]
	l1_shared_cross			= layer_set["l1"]["shared_2x2_weight_cross"]
	l1_adapt_2x2 			= layer_set["l1"]["adapt_2x2_connection"]
	l1_adapt_baseleak_share = layer_set["l1"]["adapt_share_baseleak_t"]
	l1_adapt_share_addt		= layer_set["l1"]["adapt_share_add_t"]

	num_neurons_l1 = neurons
	# Determine the number of parameters that will be trained
	if encoding_layer:
		if l1_w_diagonal:  num_neurons_l0 = neurons
		else: 			   num_neurons_l0 = layer_set["l0"]["neurons"]
	
	if encoding_layer:
		if l1_w_diagonal:
			# Encoded, Diagonal (1x1 or 2x2 diag), Shared
			if l1_shared_wb: 
				if l1_shared_cross: num_param_l1_weight= int(num_neurons_l1)
				else: num_param_l1_weight= int(num_neurons_l1/2)
					

			# Encoded, Diagonal, Not Shared
			else:
				# Encoded, Diagonal (2x2), Not Shared 			
				if l1_w_2x2:
					num_param_l1_weight = int(num_neurons_l1*2)
				# Encoded, Diagonal (1x1), Not Shared
				else: 
					num_param_l1_weight = num_neurons_l1
		else:
			# Encoded, NOT Diagonal, Shared
			if l1_shared_wb: num_param_l1_weight= int(num_neurons_l1*num_neurons_l0/4)

			# Encoded, NOT Diagonal, Not Shared
			else:			 num_param_l1_weight =int(num_neurons_l1*num_neurons_l0)
	else:
		# NOT Encoded, Shared
		if l1_shared_wb:	 num_param_l1_weight = int(num_neurons_l1/2)

		# NOT Encoded, NOT Shared
		else: 				 num_param_l1_weight = num_neurons_l1


	num_param_l1_bias 	= int(num_neurons_l1/2) if l1_shared_wb						else num_neurons_l1
	num_param_l1_leak_iv= int(num_neurons_l1/2) if layer_set["l1"]["shared_leak_iv"]	else num_neurons_l1 
	num_param_l2_wb 	= int(num_neurons_l1/2) if l2_shared_wb  					else num_neurons_l1   

	num_param_l1_addt 	= int(num_neurons_l1*2) if l1_adapt_2x2						else num_neurons_l1
	if l1_adapt_share_addt: num_param_l1_addt = int(num_neurons_l1/2)
	
	num_param_l1_base_leak_t = int(num_neurons_l1/2) if l1_adapt_baseleak_share		else num_neurons_l1
	
	init_param["l1_thres"]	= torch.ones(num_neurons_l1).float()
	init_param["l1_leak_v"]	= torch.ones(num_param_l1_leak_iv).float()
	init_param["l1_leak_i"]	= torch.ones(num_param_l1_leak_iv).float()
	init_param["l2_leak"] 	= torch.ones(1).float()
	

	if layer_set["l1"]["bias"]:
		init_param["l1_bias"]	= torch.ones(num_param_l1_bias).float()
	
	# Determine the shape of the parameters. When not shared (or not encoded for weights l1), they will be in the correct shape 
	init_param["l1_weights"]= torch.ones(num_param_l1_weight).float()		# flatten tensor, shape will be determined later
	init_param["l2_weights"]= torch.ones(num_param_l2_wb).float()		#flatten tensor, shape will be determined later
	
		
	
	# Init Adaptive parameters
	if l1_adapt:
		init_param["l1_leak_t"] = torch.ones(num_param_l1_base_leak_t).float()
		init_param["l1_base_t"] = torch.ones(num_param_l1_base_leak_t).float()
		init_param["l1_add_t"] = torch.ones(int(num_param_l1_addt)).float()

	# Init Recurrent Weights
	if l1_recur:
		if l1_recur2x2:
			init_param["l1_weights_rec"]= torch.ones(num_neurons_l1*2).float()
		else:
			init_param["l1_weights_rec"]= torch.ones(num_neurons_l1,num_neurons_l1).float()
	return init_param


def init_l0_l1_l2(neurons_l1,layer_set, init_param):
	if layer_set["l1"]["w_diagonal"]: neurons_l0 = neurons_l1
	else:  							  neurons_l0 = layer_set["l0"]["neurons"]
	
	num_param_l0_wb 	= int(neurons_l0/2) if layer_set["l0"]["shared_weight_and_bias"]  	else neurons_l0
	num_param_l0_leakiv = int(neurons_l0/2) if layer_set["l0"]["shared_leak_iv"]  			else neurons_l0
	num_param_l0_thres 	= int(neurons_l0/2) if layer_set["l0"]["shared_thres"]  			else neurons_l0

	init_param["l0_thres"]	 = torch.ones(num_param_l0_thres).float()

	init_param["l0_leak_v"]	 = torch.ones(num_param_l0_leakiv).float()
	init_param["l0_leak_i"]  = torch.ones(num_param_l0_leakiv).float()  

	init_param["l0_weights"] = torch.ones(num_param_l0_wb).float()	# Shpae (out,in)
	if layer_set["l0"]["bias"]: 
		init_param["l0_bias"]	= torch.ones(num_param_l0_wb).float()
	
	return init_param
