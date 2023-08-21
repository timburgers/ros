#!/usr/bin/env python3
# Pytorch version: https://github.com/isakbosman/pytorch_arm_builds/blob/main/torch-1.7.0a0-cp37-cp37m-linux_armv6l.whl
# https://github.com/nmilosev/pytorch-arm-builds
# Can not install noetic on pi (python3.7 and torch are installed)
# Can not install a torch for python2 (kinetic and python2.7 are installed)
"""
# TO DO: 
    1) Check publisher queue size
    2) Call PID params from launch file (parameters)
    3) self.pub_msg.angle, automate the 1 or 10 (min, max) based on sign (same for keyboard_simple)
"""
from SNN_LIF_LI_init import L1_Decoding_SNN, Encoding_L1_Decoding_SNN
import time
import rospy
import myPID as PID
import torch
import numpy as np
import pickle
from pygad.torchga import model_weights_as_dict

# Subscriber messages
# from radar_targets_msgs.msg import MyEventArray
# from radar_targets_msgs.msg import MyEvent
from std_msgs.msg import Float32
# Publishing messages
from motor_control.msg import MotorCommand
from motor_control.msg import PID_seperate
from motor_control.msg import SNN_seperate

# Global variables:
FREQUENCY = 10.0
MODE = "snn"        #either "pid" or "pid_xm" (x=[3,4]) or "pid_h" or "snn"

# Only applicable if MODE == "pid"
P = 0
I = 0
D = 0

#Only applicable if MODE == "snn" NOTE: when None is specified, the other controllers will use PID
SNN_PID = None            # Will override the P, I and D variables
SNN_PD = "655-brisk-sun" # Will override the P and D varaibles
SNN_P = None
SNN_I = None
SNN_D = None


class Controller:
    
    def __init__(self):
        self.mode = MODE
        # Subscribers and Publisher
        # self.sub_radar = rospy.Subscriber("/h_meas", MyEventArray, self.callback_radar)
        self.sub_h_meas = rospy.Subscriber("/tfmini_ros_node/TFmini", Float32, self.callback_h_meas, tcp_nodelay=True)
        self.sub_h_ref = rospy.Subscriber("/h_ref", Float32, self.callback_h_ref, tcp_nodelay=True)
        self.pub_motor = rospy.Publisher("/motor_control", MotorCommand, queue_size = 1,tcp_nodelay=True)        #send to the motor_controller
        
    
        # Messages
        self.pub_msg = MotorCommand()
        self.pub_msg_pid = Float32()
        self.pub_msg_snn = Float32()

        # Some important parameters
        self.h_meas = 0.0
        self.h_ref = 0.0
        self.range = 0.0
        self.range_filter = 0.0
        self.error = 0.0

        # Controllers
        if self.mode =="pid" or self.mode=="pid_3m" or self.mode=="pid_4m" or self.mode=="pid_h":
            self.pub_pid   = rospy.Publisher("/u_pid", PID_seperate, queue_size = 1,tcp_nodelay=True)
            self.pid = PID.PID(P, I, D, 1/FREQUENCY, True) # self.pid = PID.PID(P, I, D, dt, simple)
        
        elif self.mode == "snn":
            self.pub_snn   = rospy.Publisher("/u_snn", SNN_seperate, queue_size = 1, tcp_nodelay=True)

            self.snn_p = SNN_P
            self.snn_i = SNN_I
            self.snn_d = SNN_D

            if SNN_PD != None:
                self.snn_p = SNN_PD
                self.snn_d = SNN_PD

            if SNN_PID != None:
                self.snn_p = SNN_PID
                self.snn_i = SNN_PID 
                self.snn_d = SNN_PID

            self.SNN_FILES = [self.snn_p, self.snn_i, self.snn_d]
            self.num_snn_controller, self.snn_files = count_different_strings(self.SNN_FILES)

            if self.num_snn_controller == 1:
                self.init_SNN_model(self.snn_files[0])
            
            elif self.num_snn_controller == 2:
                self.init_SNN_model_2(self.snn_files[0], self.snn_files[1])
            
            elif self.num_snn_controller == 3:
                self.init_SNN_model_3(self.snn_files[0],self.snn_files[1],self.snn_files[2])

            # This means that atleast one of the PID parameters, need to be run without SNN and with PID
            if None in self.SNN_FILES:
                self.pub_pid   = rospy.Publisher("/u_pid", PID_seperate, queue_size = 1,tcp_nodelay=True)
                self.pid = PID.PID(P, I, D, 1/FREQUENCY, True) # self.pid = PID.PID(P, I, D, dt, simple)





    def init_SNN_model(self, pkl_file):
        # Unpack the selected .pkl file 
        pickle_in = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/10hz/"+pkl_file+".pkl","rb")
        dict_solutions = pickle.load(pickle_in)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions       = dict_solutions["test_solutions"]     #All solutions to the test sequence
        solutions_error      = dict_solutions["error"]              #All errors of the test solutions
        config               = dict_solutions["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind = np.argmin(solutions_error)
        self.solution = test_solutions[best_sol_ind+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        self.enc_lay_enabled = config["LAYER_SETTING"]["l0"]["enabled"]
        if self.enc_lay_enabled: self.controller = Encoding_L1_Decoding_SNN(None, config["NEURONS"], config["LAYER_SETTING"])
        else:              self.controller = L1_Decoding_SNN(None, config["NEURONS"], config["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters = model_weights_as_dict(self.controller, self.solution)
        self.controller.load_state_dict(final_parameters)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        if self.enc_lay_enabled: self.controller.l0.init_reshape()
        self.controller.l1.init_reshape()
        self.controller.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        if self.enc_lay_enabled: self.state_l0      = torch.zeros(self.controller.l0.neuron.state_size, 1, self.controller.l1_input)
        self.state_l1                               = torch.zeros(self.controller.l1.neuron.state_size, 1, self.controller.neurons) 
        self.state_l2                               = torch.zeros(self.controller.l2.neuron.state_size,1)

    def init_SNN_model_2(self, snn_contr_1, snn_contr_2):
        # Unpack the selected .pkl file 
        pickle_snn1 = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/10hz/"+snn_contr_1+".pkl","rb")
        dict_snn1 = pickle.load(pickle_snn1)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_snn1       = dict_snn1["test_solutions"]     #All solutions to the test sequence
        solutions_error_snn1      = dict_snn1["error"]              #All errors of the test solutions
        config_snn1               = dict_snn1["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_snn1 = np.argmin(solutions_error_snn1)
        self.solution_snn1 = test_solutions_snn1[best_sol_ind_snn1+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_snn1["LAYER_SETTING"]["l0"]["enabled"]: self.controller_snn1 = Encoding_L1_Decoding_SNN(None, config_snn1["NEURONS"], config_snn1["LAYER_SETTING"])
        else:                                           self.controller_snn1 = L1_Decoding_SNN(None, config_snn1["NEURONS"], config_snn1["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_snn1 = model_weights_as_dict(self.controller_snn1, self.solution_snn1)
        self.controller_snn1.load_state_dict(final_parameters_snn1)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_snn1.l0.init_reshape()
        self.controller_snn1.l1.init_reshape()
        self.controller_snn1.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_snn1      = torch.zeros(self.controller_snn1.l0.neuron.state_size, 1, self.controller_snn1.l1_input)
        self.state_l1_snn1      = torch.zeros(self.controller_snn1.l1.neuron.state_size, 1, self.controller_snn1.neurons) 
        self.state_l2_snn1      = torch.zeros(self.controller_snn1.l2.neuron.state_size,1)

    ### Init the second snn controller

        # Unpack the selected .pkl file 
        pickle_snn2 = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/10hz/"+snn_contr_2+".pkl","rb")
        dict_snn2 = pickle.load(pickle_snn2)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_snn2       = dict_snn2["test_solutions"]     #All solutions to the test sequence
        solutions_error_snn2      = dict_snn2["error"]              #All errors of the test solutions
        config_snn2               = dict_snn2["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_snn2 = np.argmin(solutions_error_snn2)
        self.solution_snn2 = test_solutions_snn2[best_sol_ind_snn2+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_snn2["LAYER_SETTING"]["l0"]["enabled"]: self.controller_snn2 = Encoding_L1_Decoding_SNN(None, config_snn2["NEURONS"], config_snn2["LAYER_SETTING"])
        else:                                           self.controller_snn2 = L1_Decoding_SNN(None, config_snn2["NEURONS"], config_snn2["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_snn2 = model_weights_as_dict(self.controller_snn2, self.solution_snn2)
        self.controller_snn2.load_state_dict(final_parameters_snn2)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_snn2.l0.init_reshape()
        self.controller_snn2.l1.init_reshape()
        self.controller_snn2.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_snn2      = torch.zeros(self.controller_snn2.l0.neuron.state_size, 1, self.controller_snn2.l1_input)
        self.state_l1_snn2      = torch.zeros(self.controller_snn2.l1.neuron.state_size, 1, self.controller_snn2.neurons) 
        self.state_l2_snn2      = torch.zeros(self.controller_snn2.l2.neuron.state_size,1)

    def init_SNN_model_3(self, snn_contr_1, snn_contr_2, snn_contr_3):
        # Unpack the selected .pkl file 
        pickle_snn1 = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/10hz/"+snn_contr_1+".pkl","rb")
        dict_snn1 = pickle.load(pickle_snn1)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_snn1       = dict_snn1["test_solutions"]     #All solutions to the test sequence
        solutions_error_snn1      = dict_snn1["error"]              #All errors of the test solutions
        config_snn1               = dict_snn1["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_snn1 = np.argmin(solutions_error_snn1)
        self.solution_snn1 = test_solutions_snn1[best_sol_ind_snn1+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_snn1["LAYER_SETTING"]["l0"]["enabled"]: self.controller_snn1 = Encoding_L1_Decoding_SNN(None, config_snn1["NEURONS"], config_snn1["LAYER_SETTING"])
        else:                                           self.controller_snn1 = L1_Decoding_SNN(None, config_snn1["NEURONS"], config_snn1["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_snn1 = model_weights_as_dict(self.controller_snn1, self.solution_snn1)
        self.controller_snn1.load_state_dict(final_parameters_snn1)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_snn1.l0.init_reshape()
        self.controller_snn1.l1.init_reshape()
        self.controller_snn1.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_snn1      = torch.zeros(self.controller_snn1.l0.neuron.state_size, 1, self.controller_snn1.l1_input)
        self.state_l1_snn1      = torch.zeros(self.controller_snn1.l1.neuron.state_size, 1, self.controller_snn1.neurons) 
        self.state_l2_snn1      = torch.zeros(self.controller_snn1.l2.neuron.state_size,1)

        ### INIT THE SECOND SNN CONTROLLER ###

        # Unpack the selected .pkl file 
        pickle_snn2 = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/10hz/"+snn_contr_2+".pkl","rb")
        dict_snn2 = pickle.load(pickle_snn2)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_snn2       = dict_snn2["test_solutions"]     #All solutions to the test sequence
        solutions_error_snn2      = dict_snn2["error"]              #All errors of the test solutions
        config_snn2               = dict_snn2["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_snn2 = np.argmin(solutions_error_snn2)
        self.solution_snn2 = test_solutions_snn2[best_sol_ind_snn2+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_snn2["LAYER_SETTING"]["l0"]["enabled"]: self.controller_snn2 = Encoding_L1_Decoding_SNN(None, config_snn2["NEURONS"], config_snn2["LAYER_SETTING"])
        else:                                           self.controller_snn2 = L1_Decoding_SNN(None, config_snn2["NEURONS"], config_snn2["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_snn2 = model_weights_as_dict(self.controller_snn2, self.solution_snn2)
        self.controller_snn2.load_state_dict(final_parameters_snn2)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_snn2.l0.init_reshape()
        self.controller_snn2.l1.init_reshape()
        self.controller_snn2.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_snn2      = torch.zeros(self.controller_snn2.l0.neuron.state_size, 1, self.controller_snn2.l1_input)
        self.state_l1_snn2      = torch.zeros(self.controller_snn2.l1.neuron.state_size, 1, self.controller_snn2.neurons) 
        self.state_l2_snn2      = torch.zeros(self.controller_snn2.l2.neuron.state_size,1)

        ### INIT THE THIRD SNN CONTROLLER ###

        # Unpack the selected .pkl file 
        pickle_snn3 = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/10hz/"+snn_contr_3+".pkl","rb")
        dict_snn3 = pickle.load(pickle_snn3)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_snn3       = dict_snn3["test_solutions"]     #All solutions to the test sequence
        solutions_error_snn3      = dict_snn3["error"]              #All errors of the test solutions
        config_snn3               = dict_snn3["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_snn3 = np.argmin(solutions_error_snn3)
        self.solution_snn3 = test_solutions_snn3[best_sol_ind_snn3+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_snn3["LAYER_SETTING"]["l0"]["enabled"]: self.controller_snn3 = Encoding_L1_Decoding_SNN(None, config_snn3["NEURONS"], config_snn3["LAYER_SETTING"])
        else:                                           self.controller_snn3 = L1_Decoding_SNN(None, config_snn3["NEURONS"], config_snn3["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_snn3 = model_weights_as_dict(self.controller_snn3, self.solution_snn3)
        self.controller_snn3.load_state_dict(final_parameters_snn3)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_snn3.l0.init_reshape()
        self.controller_snn3.l1.init_reshape()
        self.controller_snn3.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_snn3      = torch.zeros(self.controller_snn3.l0.neuron.state_size, 1, self.controller_snn3.l1_input)
        self.state_l1_snn3      = torch.zeros(self.controller_snn3.l1.neuron.state_size, 1, self.controller_snn3.neurons) 
        self.state_l2_snn3      = torch.zeros(self.controller_snn3.l2.neuron.state_size,1)

    def callback_h_ref(self, msg):
        self.h_ref = msg.data

    def callback_h_meas(self, msg):
        self.h_meas = msg.data

    def update_PID(self):
        if self.mode == "pid":      pe,ie,de = self.pid.update_simple(self.error)
        elif self.mode == "pid_3m": pe,ie,de = self.pid.update_simple_3m(self.error)
        elif self.mode == "pid_4m": pe,ie,de = self.pid.update_simple_4m(self.error)
        elif self.mode == "pid_h":  pe,ie,de = self.pid.update_simple_h(self.error, self.h_meas)
        else:
            pe,ie,de = self.pid.update_simple_3m(self.error)
            # pe,ie,de = self.pid.update_simple(self.error)
        
        return pe,ie,de 

    def update_SNN(self, number_of_snns):
        error = torch.Tensor([self.error])
        
        if number_of_snns==1:
            self.state_l0, self.state_l1, self.state_l2 = self.controller(error,self.state_l0, self.state_l1, self.state_l2)
            return self.state_l2, 0 , 0
        
        elif number_of_snns==2:
            self.state_l0_snn1, self.state_l1_snn1, self.state_l2_snn1 = self.controller_snn1(error,self.state_l0_snn1, self.state_l1_snn1, self.state_l2_snn1)
            self.state_l0_snn2, self.state_l1_snn2, self.state_l2_snn2 = self.controller_snn2(error,self.state_l0_snn2, self.state_l1_snn2, self.state_l2_snn2)
            return self.state_l2_snn1, self.state_l2_snn2, 0
        
        elif number_of_snns==3:
            self.state_l0_snn1, self.state_l1_snn1, self.state_l2_snn1 = self.controller_snn1(error,self.state_l0_snn1, self.state_l1_snn1, self.state_l2_snn1)
            self.state_l0_snn2, self.state_l1_snn2, self.state_l2_snn2 = self.controller_snn2(error,self.state_l0_snn2, self.state_l1_snn2, self.state_l2_snn2)
            self.state_l0_snn3, self.state_l1_snn3, self.state_l2_snn3 = self.controller_snn3(error,self.state_l0_snn3, self.state_l1_snn3, self.state_l2_snn3)
            return self.state_l2_snn1, self.state_l2_snn2, self.state_l2_snn3

    def update_command(self):
        # rospy.loginfo("h_meas = " + str(self.h_meas))
        self.error = self.h_ref - self.h_meas
        self.h_ref_used = self.h_ref
        self.h_meas_used = self.h_meas
        # rospy.loginfo("error = "+ str(self.error))

        # Create motor command from PID
        if self.mode == "pid" or self.mode == "pid_3m" or self.mode == "pid_4m" or self.mode =="pid_h":
            pe,ie,de  = self.update_PID()
            self.pub_msg_pid = PID_seperate()
            self.pub_msg_pid.pe = pe
            self.pub_msg_pid.ie = ie
            self.pub_msg_pid.de = de
            self.pub_msg_pid.meas = self.h_meas_used
            self.pub_msg_pid.ref = self.h_ref_used
            u = pe + ie + de
            # for more insight in pid
            self.pub_pid.publish(self.pub_msg_pid)

        elif self.mode == "snn":
            self.pub_msg_snn = SNN_seperate()

            u1, u2, u3 = self.update_SNN(self.num_snn_controller)
            u = u1 + u2 + u3

            ### Create SNN msgs
            #Case 1) SNN is full PID
            if SNN_PID !=None:
                self.pub_msg_snn.snn_pid = u1

            #Case 2) SNN is PD and optionally also I
            if SNN_PD != None:
                self.pub_msg_snn.snn_pd = u1
                if SNN_I != None:
                    self.pub_msg_snn.snn_i = u2
            
            #Case 3) SNN is one or more of the PID seperately
            if SNN_PID ==None and SNN_PD== None:
                u_list = [u1,u2,u3]
                ind_snn_cont =0
                for index, item in enumerate(self.SNN_FILES):
                    if item !=None and index ==0:
                        self.pub_msg_snn.snn_p = u_list[ind_snn_cont]
                        ind_snn_cont +=1

                    elif item !=None and index ==1:
                        self.pub_msg_snn.snn_i = u_list[ind_snn_cont]
                        ind_snn_cont +=1
                    
                    elif item !=None and index ==2:
                        self.pub_msg_snn.snn_d = u_list[ind_snn_cont]
                        ind_snn_cont +=1
            self.pub_snn.publish(self.pub_msg_snn)

            # If there is also a PID required
            if None in self.SNN_FILES:
                pe,ie,de  = self.update_PID()
                self.pub_msg_pid = PID_seperate()
                self.pub_msg_pid.meas = self.h_meas_used
                self.pub_msg_pid.ref = self.h_ref_used  
                
                for index, item in enumerate(self.SNN_FILES):
                    if item is None and index ==0:
                        u += pe
                        self.pub_msg_pid.pe = pe

                    elif item is None and index ==1:
                        u += ie
                        self.pub_msg_pid.ie = ie
                    
                    elif item is None and index ==2:
                        u += de
                        self.pub_msg_pid.de = de
                self.pub_msg_pid.de = de
                self.pub_pid.publish(self.pub_msg_pid)

        rospy.loginfo("snn msg = "+ str(self.pub_msg_snn))
        rospy.loginfo("pid msg = "+ str(self.pub_msg_pid))
            

        #Create message for the motor controller
        # self.pub_msg.ts = rospy.get_rostime()

        if u >= 0:
            self.pub_msg.angle = 10
        else:
            u = u*(-1)
            self.pub_msg.angle = 1

        self.pub_msg.cw_speed = u
        self.pub_msg.ccw_speed = u

        self.pub_motor.publish(self.pub_msg)

def count_different_strings(lst):
    unique_strings = set()
    
    for item in lst:
        if item is not None and isinstance(item, str):
            unique_strings.add(item)
    
    return len(unique_strings), list(unique_strings)


if __name__ == '__main__':
    rospy.init_node('controller') # Node initialization #, anonymous=True)
    myController = Controller()   # Instantiation of the Controller class
    #rospy.spin()
    r = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        myController.update_command()
        r.sleep()

