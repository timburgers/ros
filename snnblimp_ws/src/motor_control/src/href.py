#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

FREQUENCY = 1.0

if __name__ == '__main__':
    rospy.loginfo("Generate h ref node has been started")
    rospy.init_node("generate_h_ref")
    rospy.loginfo("Generate h ref node has been started")

    pub_h_ref = rospy.Publisher("/h_ref",Float32, queue_size=5)

    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        msg=Float32()
        msg.data= 1.0
        pub_h_ref.publish(msg)
        rate.sleep()

