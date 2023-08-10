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
MODE = "pid_3m"        #either "pid" or "pid_3m" or "pid_4d" or "pid_h" or "snn" or "snn_sep" or "snn_pid"

# Only applicable if MODE == "pid"
P = 10
I = 0.75
D = 12

#Only applicable if MODE == "snn"
SNN_FULL = "271-prime-bee"

#Only applicable if MODE == "ssn_sep" or "snn_pid" NOTE: when using snn_pid, know which one to use as snn and as pid in code
SNN_PD = "228-dashing-meadow"
SNN_I = "247-desert-snowflake"


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
        if self.mode =="pid" or self.mode=="pid_3m" or self.mode=="pid_4d" or self.mode=="pid_h":
            self.pub_pid   = rospy.Publisher("/u_pid", PID_seperate, queue_size = 1,tcp_nodelay=True)
            self.pid = PID.PID(P, I, D, 1/FREQUENCY, True) # self.pid = PID.PID(P, I, D, dt, simple)
        
        elif self.mode == "snn":
            self.pub_snn   = rospy.Publisher("/u_snn", Float32, queue_size = 1, tcp_nodelay=True)
            self.init_SNN_model(SNN_FULL)

        elif self.mode == "snn_sep":
            self.pub_snn   = rospy.Publisher("/u_snn", SNN_seperate, queue_size = 1, tcp_nodelay=True)
            self.init_SNN_model_sep(SNN_PD, SNN_I)

        elif self.mode == "snn_pid":
            self.pub_snn   = rospy.Publisher("/u_snn", SNN_seperate, queue_size = 1,tcp_nodelay=True)
            self.pub_pid   = rospy.Publisher("/u_pid", PID_seperate, queue_size = 1,tcp_nodelay=True)
            self.pid = PID.PID(10, 0.75, 12, 1/FREQUENCY, True) # self.pid = PID.PID(P, I, D, dt, simple)
            self.init_SNN_model(SNN_FULL)





    def init_SNN_model(self, pkl_file):
        # Unpack the selected .pkl file 
        pickle_in = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/"+pkl_file+".pkl","rb")
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

    def init_SNN_model_sep(self, pd_contr, i_contr):
        # Unpack the selected .pkl file 
        pickle_pd = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/"+pd_contr+".pkl","rb")
        dict_pd = pickle.load(pickle_pd)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_pd       = dict_pd["test_solutions"]     #All solutions to the test sequence
        solutions_error_pd      = dict_pd["error"]              #All errors of the test solutions
        config_pd               = dict_pd["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_pd = np.argmin(solutions_error_pd)
        self.solution_pd = test_solutions_pd[best_sol_ind_pd+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_pd["LAYER_SETTING"]["l0"]["enabled"]: self.controller_pd = Encoding_L1_Decoding_SNN(None, config_pd["NEURONS"], config_pd["LAYER_SETTING"])
        else:                                           self.controller_pd = L1_Decoding_SNN(None, config_pd["NEURONS"], config_pd["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_pd = model_weights_as_dict(self.controller_pd, self.solution_pd)
        self.controller_pd.load_state_dict(final_parameters_pd)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_pd.l0.init_reshape()
        self.controller_pd.l1.init_reshape()
        self.controller_pd.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_pd      = torch.zeros(self.controller_pd.l0.neuron.state_size, 1, self.controller_pd.l1_input)
        self.state_l1_pd      = torch.zeros(self.controller_pd.l1.neuron.state_size, 1, self.controller_pd.neurons) 
        self.state_l2_pd      = torch.zeros(self.controller_pd.l2.neuron.state_size,1)

    ### Init the I controller

        # Unpack the selected .pkl file 
        pickle_i = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/snn_controllers/"+i_contr+".pkl","rb")
        dict_i = pickle.load(pickle_i)

        #Unpack in usefull variables
        # solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
        test_solutions_i       = dict_i["test_solutions"]     #All solutions to the test sequence
        solutions_error_i      = dict_i["error"]              #All errors of the test solutions
        config_i               = dict_i["config"]             #Configfile set for the evolution of the network

        # Select the best performing (lowest test error) in the network
        best_sol_ind_i = np.argmin(solutions_error_i)
        self.solution_i = test_solutions_i[best_sol_ind_i+1] #+1 since first of solution_testrun is only zeros

        # Initiailize the SNN model (only structure, not weights)
        if config_i["LAYER_SETTING"]["l0"]["enabled"]: self.controller_i = Encoding_L1_Decoding_SNN(None, config_i["NEURONS"], config_i["LAYER_SETTING"])
        else:                                           self.controller_i = L1_Decoding_SNN(None, config_i["NEURONS"], config_i["LAYER_SETTING"])

        # Convert the model weight in a dict 
        final_parameters_i = model_weights_as_dict(self.controller_i, self.solution_i)
        self.controller_i.load_state_dict(final_parameters_i)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        self.controller_i.l0.init_reshape()
        self.controller_i.l1.init_reshape()
        self.controller_i.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        self.state_l0_i      = torch.zeros(self.controller_i.l0.neuron.state_size, 1, self.controller_i.l1_input)
        self.state_l1_i      = torch.zeros(self.controller_i.l1.neuron.state_size, 1, self.controller_i.neurons) 
        self.state_l2_i      = torch.zeros(self.controller_i.l2.neuron.state_size,1)


    def callback_h_ref(self, msg):
        self.h_ref = msg.data

    def callback_h_meas(self, msg):
        self.h_meas = msg.data

    def update_PID(self):
        if self.mode == "pid":      pe,ie,de = self.pid.update_simple(self.error)
        elif self.mode == "pid_3m": pe,ie,de = self.pid.update_simple_3m(self.error)
        elif self.mode == "pid_4d": pe,ie,de = self.pid.update_simple_4d(self.error)
        elif self.mode == "pid_h":  pe,ie,de = self.pid.update_simple_h(self.error, self.h_meas)
        
        return pe,ie,de 

    def update_SNN(self):
        error = torch.Tensor([self.error])
        
        if self.mode == "snn" or self.mode == "snn_pid":
            if self.enc_lay_enabled:      self.state_l0, self.state_l1, self.state_l2 = self.controller(error,self.state_l0, self.state_l1, self.state_l2)
            else:                         self.state_l1, self.state_l2 = self.controller(error, self.state_l1, self.state_l2)
            return self.state_l2
        
        elif self.mode == "snn_sep":
            self.state_l0_pd, self.state_l1_pd, self.state_l2_pd = self.controller_pd(error,self.state_l0_pd, self.state_l1_pd, self.state_l2_pd)
            self.state_l0_i, self.state_l1_i, self.state_l2_i = self.controller_i(error,self.state_l0_i, self.state_l1_i, self.state_l2_i)
            return self.state_l2_pd, self.state_l2_i

    def update_command(self):
        rospy.loginfo("h_meas = " + str(self.h_meas))
        self.error = self.h_ref - self.h_meas
        
        # Create motor command from PID
        if self.mode == "pid" or self.mode == "pid_3m" or self.mode == "pid_4d" or self.mode =="pid_h":
            pe,ie,de  = self.update_PID()
            self.pub_msg_pid = PID_seperate()
            self.pub_msg_pid.pe = pe
            self.pub_msg_pid.ie = ie
            self.pub_msg_pid.de = de
            u = pe + ie + de
            # for more insight in pid
            self.pub_pid.publish(self.pub_msg_pid)

        elif self.mode == "snn":
        # Create motor command from SNN
            u = self.update_SNN()
            # u =-u
            self.pub_msg_snn = u
            self.pub_snn.publish(self.pub_msg_snn)

        elif self.mode == "snn_sep":
        # Create motor command from SNN
            u_pd, u_i = self.update_SNN()
            # u =-u
            self.pub_msg_snn = SNN_seperate()
            self.pub_msg_snn.snn_pd = u_pd
            self.pub_msg_snn.snn_i = u_i
            self.pub_snn.publish(self.pub_msg_snn)
            u = u_pd  + u_i
        

        elif self.mode == "snn_pid":
        # Create motor command from SNN
            u_snn = self.update_SNN()
            pe,ie,de  = self.update_PID()
            u = u_snn + ie
            
            self.pub_msg_pid = PID_seperate()
            self.pub_msg_pid.ie = ie
            self.pub_pid.publish(self.pub_msg_pid)

            self.pub_msg_snn = SNN_seperate()
            self.pub_msg_snn.snn_pd = u_snn             #either "snn_pd" or "snn_i"
            self.pub_snn.publish(self.pub_msg_snn)



        #Create message for the motor controller
        self.pub_msg.ts = rospy.get_rostime()

        if u >= 0:
            self.pub_msg.angle = 10
        else:
            u = u*(-1)
            self.pub_msg.angle = 1

        self.pub_msg.cw_speed = u
        self.pub_msg.ccw_speed = u

        self.pub_motor.publish(self.pub_msg)



if __name__ == '__main__':
    rospy.init_node('controller') # Node initialization #, anonymous=True)
    myController = Controller()   # Instantiation of the Controller class
    #rospy.spin()
    r = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        myController.update_command()
        r.sleep()

