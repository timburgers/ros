#!/usr/bin/env python3
import rospy
import numpy as np
from motor_control.msg import OptitrackPose
from natnet_client import NatNetClient

class OptitrackPublisher(object):
    def __init__(self):
        rospy.init_node("optitrack_publisher")
        self.publisher = rospy.Publisher("/optitrack", OptitrackPose, queue_size=10)
        # TODO: remove hardcoded id, use ros parameter or args or whatever
        self.ot_id = 1
        # start client in separate thread
        streaming_client = NatNetClient()
        streaming_client.newFrameListener = self.receive_new_frame
        streaming_client.rigidBodyListener = self.receive_rigidbody_frame
        streaming_client.run()
        timer_period = 0.02  # seconds TODO: is this enough?
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.publish_cb)
        # global vars
        self.p_ned = None
        self.q_ned = None
        self.tracking_valid = False
        # TODO: use px4 timesync?
        self.prev_t = rospy.Time.now()

    def publish_cb(self, event):
        msg = OptitrackPose()
        if self.tracking_valid and (self.p_ned is not None and rospy.Time.now() - self.prev_t <= rospy.Duration(0.5)):
            msg.p_w = self.p_ned
            msg.q_wb = self.q_ned
            msg.tracking_valid = True
            # TODO: remove?
            rospy.loginfo(f"publishing NED position (xyz): {self.p_ned[0]:.2f}, {self.p_ned[1]:.2f}, {self.p_ned[2]:.2f}")
            rospy.loginfo(
                f"publishing NED attitude (wxyz): {self.q_ned[0]:.2f}, {self.q_ned[1]:.2f}, {self.q_ned[2]:.2f}, {self.q_ned[3]:.2f}"
            )
        if rospy.Time.now() - self.prev_t > rospy.Duration(0.5):
            self.tracking_valid = False
            msg.tracking_valid = self.tracking_valid
            rospy.loginfo("lost optitrack")
        self.publisher.publish(msg)

    def receive_rigidbody_frame(self, id, position, rotation, t_valid):
        # check id
        if t_valid and (id == self.ot_id):
            # register position
            self.p_ned = self.p_ot2ned(position)
            # register attitude as quaternion
            self.q_ned = self.q_ot2ned(rotation)
            self.tracking_valid = t_valid
            self.prev_t = rospy.Time.now()

    def receive_new_frame(self, *args, **kwargs):
        pass

    @staticmethod
    def p_ot2ned(p_ot):
        # convert position from optitrack to NED
        p_ned = np.zeros(3, dtype=np.float32)
        p_ned[0] = p_ot[2]  # NED.x = OT.z
        p_ned[1] = -p_ot[0]  # NED.y = -OT.x
        p_ned[2] = -p_ot[1]  # NED.z = -OT.y
        return p_ned

    @staticmethod
    def q_ot2ned(q_ot):
        # convert attitude quaternion from optitrack to NED
        q_ned = np.zeros(4, dtype=np.float32)
        q_ned[0] = q_ot[3]  # NED.w = OT.w but change to first
        q_ned[1] = q_ot[2]  # NED.x = OT.z
        q_ned[2] = -q_ot[0]  # NED.y = -OT.x
        q_ned[3] = -q_ot[1]  # NED.z = -OT.y
        return q_ned

def main():
    optitrack_publisher = OptitrackPublisher()
    rospy.spin()

if __name__ == "__main__":
    main()