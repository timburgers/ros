#!/bin/bash

export ROS_IP=192.168.12.35
export ROS_MASTER_URI=http://192.168.12.35:11311
#export ROSLAUNCH_SSH_UNKNOWN=1

source /opt/ros/noetic/setup.bash
source /home/pi/ros/snnblimp_ws/devel/setup.bash

exec "$@"
