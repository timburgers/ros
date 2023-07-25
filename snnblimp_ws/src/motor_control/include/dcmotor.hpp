#ifndef DCMOTORS_HPP
#define DCMOTORS_HPP

// Compile with:
// g++ -Wall -Wextra -Werror -o test dcmotors.cpp -lwiringPi

// Including necessary libraries
#include <wiringPi.h>
#include <iostream>
#include <ros/ros.h>
#include <stdlib.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
#include "motor_control/MotorCommand.h"

class Motor{

private:

    // Attributes
    float _cw_pwmPin;
    int _cw_dirPin;
    float _ccw_pwmPin;
    int _ccw_dirPin;
    int _max_speed;
    float _speed;

    void init_io();
    void correctSpeed(float &speed);

    // ROS Attributes
    ros::NodeHandle nh;
    ros::Subscriber speed_sub;
    //std_msgs::Int32 speed_msg;

    ros::Subscriber dcmotor_alive_sub_;
    ros::Time last_received_time_;
    ros::Timer check_timer_;

public:
    Motor();
    void setSpeed(const motor_control::MotorCommand& msg);
    void dcmotorAliveCallback(const std_msgs::Bool::ConstPtr& msg);
    void checkForTimeout(const ros::TimerEvent&);

};

#endif
