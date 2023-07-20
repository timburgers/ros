#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from random import uniform
import numpy as np
import time

MODE = "list"       # either "list" or "random"

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

if __name__ == '__main__':
    rospy.init_node("generate_h_ref")
    rospy.loginfo("Generate h ref node has been started")

    pub_h_ref = rospy.Publisher("/h_ref",Float32, queue_size=10)

    if MODE == "list":
        # Parameters
        frequency = 0.2                     # [Hz]
        h_ref_list = [0.5,-1,-1.5,2,2.5,3]    

        ### running Node
        ind = 0
        rate = rospy.Rate(frequency)
        while not rospy.is_shutdown():
            msg=Float32()
            msg.data=-h_ref_list[ind]
            pub_h_ref.publish(msg)
            rate.sleep()
            if ind >= len(h_ref_list)-1:
                pass
            else: ind +=1
    
    if MODE == "random":
        # Parameters
        sim_time = "00:00:30"
        height_bounds       = [0.5,   3]    # [m]
        frequency_bounds    = [0.1, 0.2]    # [Hz]
        minimal_step_size   = 0.5           # [m]
        init_h_ref          = 1.1
        init_freq           = 0.5

        ### running Node
        freq = init_freq
        h_ref = init_h_ref

        sim_time = get_sec(sim_time)
        time_start = time.time()
        time_new =time.time()

        
        time.sleep(1)
        while (not rospy.is_shutdown() and time_new-time_start < sim_time):
            rate = rospy.Rate(freq)
            msg=Float32()
            msg.data= h_ref
            pub_h_ref.publish(msg)
            rate.sleep()

            # Get new h_ref and freq
            h_ref_prev = h_ref
            freq = round(uniform(frequency_bounds[0],frequency_bounds[1]),3)
            h_ref = round(uniform(height_bounds[0], height_bounds[1]),1)

            i=0
            time_new = time.time()
            while (abs(h_ref-h_ref_prev)<minimal_step_size and i<1000):
                h_ref = round(uniform(height_bounds[0], height_bounds[1]),1)
                i+=1
            rospy.loginfo("time to next ref point = " + str(round(1/freq,1)) + " sec" )

            if i ==1000:
                rospy.loginfo("Kept looping to find new reference withing given parameters")
                rospy.signal_shutdown("Kept looping to find new reference withing given parameters")
            
        if time_new-time_start > sim_time:
            rospy.loginfo("Time limit exceeded for href, shutdown node")
            rospy.signal_shutdown("Time limit exceeded for href, shutdown node")
