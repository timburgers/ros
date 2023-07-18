#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

FREQUENCY = .2

if __name__ == '__main__':
    rospy.init_node("generate_h_ref")
    rospy.loginfo("Generate h ref node has been started")

    pub_h_ref = rospy.Publisher("/h_ref",Float32, queue_size=10)

    # h_ref_list = [0.1,0.2,0.3,0.4,0.7,0.8,0.9,1.0,1.1]
    h_ref_list = [0.5,1,1.5,2,2.5,3]
    ind = 0

    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        msg=Float32()
        msg.data=-h_ref_list[ind]
        pub_h_ref.publish(msg)
        rate.sleep()
        if ind >= len(h_ref_list)-1:
            pass
        else: ind +=1

