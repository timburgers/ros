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

# Global variables:
FREQUENCY = 30.0
FILENAME = "345-morning-tree"

class Controller:
    
    def __init__(self):
        # Subscribers and Publisher
        # self.sub_radar = rospy.Subscriber("/h_meas", MyEventArray, self.callback_radar)
        self.sub_h_meas = rospy.Subscriber("/tfmini_ros_node/TFmini", Float32, self.callback_h_meas, tcp_nodelay=True)
        self.sub_h_ref = rospy.Subscriber("/h_ref", Float32, self.callback_h_ref, tcp_nodelay=True)
        self.pub_motor = rospy.Publisher("/motor_control", MotorCommand, queue_size = 1,tcp_nodelay=True)        #send to the motor_controller
        self.pub_pid   = rospy.Publisher("/u_pid", Float32, queue_size = 1,tcp_nodelay=True)
        self.pub_snn   = rospy.Publisher("/u_snn", Float32, queue_size = 1, tcp_nodelay=True)

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
        self.pid = PID.PID(6, 0.3, 1.5, 1/FREQUENCY, True) # self.pid = PID.PID(P, I, D, dt, simple)
        
        # SNN
        # self.init_SNN_model()


    def init_SNN_model(self):
        # Unpack the selected .pkl file 
        pickle_in = open("/home/pi/ros/snnblimp_ws/src/motor_control/src/"+FILENAME+".pkl","rb")
        dict_solutions = pickle.load(pickle_in)

        #Unpack in usefull variables
        solution             = dict_solutions["best_solution"]      #Lowest error (Overall, not only test sequence)
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
        final_parameters = model_weights_as_dict(self.controller, solution)
        self.controller.load_state_dict(final_parameters)

        # Assign the complete network its paramaeters (since some parameters are shared during training)
        if self.enc_lay_enabled: self.controller.l0.init_reshape()
        self.controller.l1.init_reshape()
        self.controller.l2.init_reshape()

        #Initialize the states of all the neurons in each layer to zero
        if self.enc_lay_enabled: self.state_l0      = torch.zeros(self.controller.l0.neuron.state_size, 1, self.controller.l1_input)
        self.state_l1                               = torch.zeros(self.controller.l1.neuron.state_size, 1, self.controller.neurons) 
        self.state_l2                               = torch.zeros(self.controller.l2.neuron.state_size,1)


    def callback_h_ref(self, msg):
        self.h_ref = msg.data

    def callback_h_meas(self, msg):
        self.h_meas = msg.data

    def update_PID(self):
        u = self.pid.update_simple(self.error)
        return u

    def update_SNN(self):
        error = torch.Tensor([self.error])
        
        if self.enc_lay_enabled:      self.state_l0, self.state_l1, self.state_l2 = self.controller(error,self.state_l0, self.state_l1, self.state_l2)
        else:                         self.state_l1, self.state_l2 = self.controller(error, self.state_l1, self.state_l2)

        return self.state_l2

    def update_command(self):
        rospy.loginfo("h_meas = " + str(self.h_meas))
        self.error = self.h_ref - self.h_meas
        
        # Create motor command from PID
        u = self.update_PID()
        self.pub_msg_pid = u
        self.pub_pid.publish(self.pub_msg_pid)

        # Create motor command from SNN
        # u = self.update_SNN()
        # self.pub_msg_snn = u
        # self.pub_snn.publish(self.pub_msg_snn)

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

