#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
# This is required since the Dcmotors file is initialised using sudo E (https://answers.ros.org/question/296295/rosok-return-true-with-dead-roscore-in-node-started-as-root/)
FREQUENCY = 2

if __name__ == '__main__':
    rospy.init_node("dc_motor_alvive")

    pub_dc_alive = rospy.Publisher("/dc_motor_alvive",Float32, queue_size=10)

    rate = rospy.Rate(FREQUENCY)

    while not rospy.is_shutdown():
        msg=Float32()
        msg=1
        pub_dc_alive.publish(msg)
        rate.sleep()


