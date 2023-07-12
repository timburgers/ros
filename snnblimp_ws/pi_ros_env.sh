#!/bin/bash

export ROS_IP=mypi
export ROS_MASTER_URI=http://mycomp:11311
#export ROSLAUNCH_SSH_UNKNOWN=1

source /home/pi/ros/snnblimp_ws/devel/setup.bash

exec "$@"
