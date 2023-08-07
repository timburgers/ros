#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from random import uniform
import numpy as np
import time
from motor_control.msg import MotorCommand
import roslaunch


MODE = "list"       # either "list" or "random"

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)





if __name__ == '__main__':
    def callback_motor_command(msg):
        motor_running = True

    rospy.init_node("generate_h_ref")
    rospy.loginfo("Generate h ref node has been started")
    pub_h_ref = rospy.Publisher("/h_ref",Float32, queue_size=1)




    if MODE == "list":
        # Parameters
        frequency = 0.04                     # [Hz]
        h_ref_list = [0.8,1.6,0.8,1.6,0.8]    

        ### running Node
        ind = 0
        rospy.wait_for_message("/motor_control", MotorCommand, timeout=None)
        time.sleep(1)
        rate = rospy.Rate(frequency)

        while not rospy.is_shutdown():
            msg=Float32()
            msg.data=h_ref_list[ind]
            pub_h_ref.publish(msg)
            rate.sleep()
            if ind >= len(h_ref_list)-1:
                rospy.signal_shutdown("Time Over")
            else: ind +=1
    


    if MODE == "random":
        # Parameters
        sim_time = "00:06:00"
        height_bounds       = [0.5,   2.2]    # [m]
        frequency_bounds    = [25, 35]      # [s]
        minimal_step_size   = 0.3           # [m]
        maximal_step_size   = 0.8           # [m]
        init_h_ref          = 0.8
        init_freq           = 0.04

        ### running Node
        freq = init_freq
        h_ref = init_h_ref

        sim_time = get_sec(sim_time)
        time_start = time.time()
        time_new =time.time()

        rospy.wait_for_message("/motor_control", MotorCommand, timeout=None)
        time.sleep(1)
        while (not rospy.is_shutdown() and time_new-time_start < sim_time):
            rate = rospy.Rate(freq)
            msg=Float32()
            msg.data= h_ref
            pub_h_ref.publish(msg)
            rate.sleep()

            # Get new h_ref and freq
            h_ref_prev = h_ref
            freq = round(uniform(1/frequency_bounds[0],1/frequency_bounds[1]),3)
            h_ref = round(uniform(height_bounds[0], height_bounds[1]),1)

            i=0
            time_new = time.time()
            while ((abs(h_ref-h_ref_prev)<minimal_step_size or abs(h_ref-h_ref_prev)>maximal_step_size )and i<2000):
                h_ref = round(uniform(height_bounds[0], height_bounds[1]),1)
                i+=1
            rospy.loginfo("time to next ref point = " + str(round(1/freq,1)) + " sec" )

            if i ==1000:
                rospy.loginfo("Kept looping to find new reference withing given parameters")
                rospy.signal_shutdown("Kept looping to find new reference withing given parameters")
            
        if time_new-time_start > sim_time:
            rospy.loginfo("Time limit exceeded for href, shutdown node")
            rospy.signal_shutdown("Time limit exceeded for href, shutdown node")
