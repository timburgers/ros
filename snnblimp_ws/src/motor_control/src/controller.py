#!/usr/bin/env python2

"""
# TO DO: 
    1) Check publisher queue size
    2) Call PID params from launch file (parameters)
    3) self.pub_msg.angle, automate the 1 or 10 (min, max) based on sign (same for keyboard_simple)
"""

import time
import rospy
import myPID as PID
# Subscriber messages
from radar_targets_msgs.msg import MyEventArray
from radar_targets_msgs.msg import MyEvent
from std_msgs.msg import Float32
# Publishing messages
from motor_control.msg import MotorCommand

# Global variables:
FREQUENCY = 3.0

class Controller:
    
    def __init__(self):

        # Subscribers and Publisher
        # self.sub_radar = rospy.Subscriber("/radar_filter", MyEventArray, self.callback_radar)
        self.sub_h_meas = rospy.Subscriber("/h_meas", Float32, self.callback_h_meas)
        self.sub_h_ref = rospy.Subscriber("/h_ref", Float32, self.callback_h_ref)
        self.pub_motor = rospy.Publisher("/motor_control", MotorCommand, queue_size = 1)

        # Messages
        self.pub_msg = MotorCommand()

        # Some important parameters
        self.h_ref = 0.0
        self.range = 0.0
        self.range_filter = 0.0
        self.error = 0.0

        # Controllers
        self.pid = PID.PID(0.8735822376592428, 0.061441163793195155, 0.0028269842479982966, 0.0333333333, True) # self.pid = PID.PID(P, I, D, dt, simple)

    def callback_h_ref(self, msg):
        self.h_ref = msg.data

    def callback_h_meas(self, msg):
        pass
        # """
        # Assuming that there's ONLY 1 TARGET
        # """
        # tmp_filter = msg.range_filter
        # tmp_range  = msg.target_events

        # if len(tmp_filter) != 0:
        #     self.range_filter = tmp_filter[0]
        # #self.range = tmp_range[0].range

    def update_command(self):
        
        self.error = self.h_ref - 0 #self.range_filter
        u = self.pid.update_simple(self.error)

        self.pub_msg.ts = rospy.get_rostime()

        self.pub_msg.cw_speed = u
        self.pub_msg.ccw_speed = u

        if u >= 0:
            self.pub_msg.angle = 1
        else:
            self.pub_msg.angle = 10

        self.pub_motor.publish(self.pub_msg)
    
    # def shut_down_motor(self):
    #     self.pub_msg.ts = rospy.get_rostime()
    #     self.pub_msg.cw_speed = 0
    #     self.pub_msg.ccw_speed = 0
    #     self.pub_msg.angle = 1
    #     self.pub_motor.publish(self.pub_msg)


    #def convert_command(self,u):
    #    pass

if __name__ == '__main__':
    rospy.init_node('controller') # Node initialization #, anonymous=True)
    myController = Controller()   # Instantiation of the Controller class
    #rospy.spin()
    r = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        myController.update_command()
        r.sleep()
    # myController.shut_down_motor()
    