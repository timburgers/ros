#include "dcmotor.hpp"
#include <iostream>
#include <ros/ros.h>
#include <stdlib.h>
// #include <signal.h>

using namespace std;

// g++ -Wall -Wextra -Werror -o test dcmotors.cpp -lwiringPi

/// Global variables
int MAX_SPEED = 10;
int CLOCK = 2;
int PWM_RANGE = 45; // 19.2 MHz / 2 / 100 = 20 kHz (https://cdn-shop.adafruit.com/datasheets/DRV2605.pdf)

/// GPIOs initialization
void Motor::init_io()
{
    wiringPiSetupGpio();
    pinMode(_cw_pwmPin,PWM_OUTPUT);
    pinMode(_ccw_pwmPin,PWM_OUTPUT);

    pwmSetMode(PWM_MODE_MS);
    pwmSetRange(PWM_RANGE);
    pwmSetClock(CLOCK);

    pinMode(_cw_dirPin,OUTPUT);
    pinMode(_ccw_dirPin,OUTPUT);
}

/**
 * Apply speed correction: from speed [0-10] to speed [9-39] (due to the 45 pwm_range)
 *
 * @param[in] speed:  speed selected by user (between 0, min, and 10, max)
 * @param[out] speed: corrected speed (between 20 and 86) corresponding to the PWM duty cycle % to be send to the driver
 */
void Motor::correctSpeed(float &speed){

    // if(speed < 0.9){
    //     speed = 0;
    // }
    // else{
    //     speed = 6.375 + 3.625 * speed;
    // }

    if(speed < 0.3){
        speed = 0;
    }
    else{
        speed = 9 + 3*speed;
    }
    
    /*
    switch (speed)
    {
    case 0 : speed = 0;  break;
    case 1 : speed = 10; break;//25; break;
    case 2 : speed = 13; break;//32; break;
    case 3 : speed = 17; break;//39; break;
    case 4 : speed = 20; break;//45; break;
    case 5 : speed = 23; break;//52; break;
    case 6 : speed = 27; break;//59; break;
    case 7 : speed = 30; break;//66; break;
    case 8 : speed = 33; break;//72; break;
    case 9 : speed = 39; break;//79; break;
    //case 10: speed = 0;  break;//40; break;//86; break;
    }
    */
}

/**
 * Motor object initialization: default constructor.  
 * CAREFUL!! -> Each DC motor should be connected to the different PWMx or they will spin in the same direction!! (keep reading) 
 * NOTE: There are only two hardware PWM signals on the Raspberry Pi:  
 * PWM0: GPIOs 13, 19 and PWM1: GPIOs 12 and 18.
 *
 * @param[in] pwmPin: PWM signal control pin definition (either 13, 19 [PWM0] or 12, 18 [PWM1])
 * @param[in] dirPin: spinning direction control pin definition
 * @param[out] _pwmPin: PWM signal control pin definition assigned to _pwmPin attribute
 * @param[out] _dirPin: spinning direction control pin definition assigned to _pwmDir attribute
 * @param[out] _max_speed: maximum speed defined for the user assigned to _max_speed attribute
 */
Motor::Motor()
{
    bool cw_pwmPin_ok = ros::param::get("~cw_pwmPin", _cw_pwmPin);
    bool cw_dirPin_ok = ros::param::get("~cw_dirPin", _cw_dirPin);
    bool ccw_pwmPin_ok = ros::param::get("~ccw_pwmPin", _ccw_pwmPin);
    bool ccw_dirPin_ok = ros::param::get("~ccw_dirPin", _ccw_dirPin);
    _max_speed = MAX_SPEED;
    init_io();

    this->speed_sub = nh.subscribe("/motor_control",1,&Motor::setSpeed,this);

    dcmotor_alive_sub_ = nh.subscribe("/dc_motor_alvive", 1, &Motor::dcmotorAliveCallback, this);
    last_received_time_ = ros::Time::now();
    check_timer_ = nh.createTimer(ros::Duration(0.3), &Motor::checkForTimeout, this);
}

void Motor::dcmotorAliveCallback(const std_msgs::Bool::ConstPtr& msg)
    {
        // Update the last received time to the current time
        last_received_time_ = ros::Time::now();
    }


void Motor::checkForTimeout(const ros::TimerEvent&)
    {
        // Calculate the time since the last message was received
        ros::Duration time_since_last_msg = ros::Time::now() - last_received_time_;
        ROS_INFO_STREAM("time checked");

        // Check if the time since the last message is greater than 1 second
        if (time_since_last_msg.toSec() > 2.0)
        {
            digitalWrite(_cw_dirPin, 0);
            pwmWrite(_cw_pwmPin, 0);
            digitalWrite(_ccw_dirPin, 0);
            pwmWrite(_ccw_pwmPin, 0);
            ROS_INFO_STREAM("Killed motor since not receiving message");
            ros::shutdown();
            
        }
    }

/**
 * Set speed and direction at which the DC motor will rotate (user speed interval: [0-10])
 *
 * @param[in] speed:  speed selected by user (between 0, min, and 10, max)
 * @param[out] speed: appropriate value of the speed to be sent to the DC motor [25-86]
 */
void Motor::setSpeed(const motor_control::MotorCommand& msg)
{
    //ROS_INFO_STREAM("speed_msg = " << speed_msg.data);

        int cw_dir_value;
        float cw_speed = msg.cw_speed;
        int ccw_dir_value;
        float ccw_speed = msg.ccw_speed;

    if (msg.cw_speed < 0)
    {
        cw_speed = -msg.cw_speed;
        cw_dir_value = 1;
    }

    else
    {
        cw_dir_value = 0;
    }

    if (msg.cw_speed > _max_speed)
    {
        cw_speed = _max_speed;
    }

    if (msg.ccw_speed < 0)
    {
        ccw_speed = -msg.ccw_speed;
        ccw_dir_value = 1;
    }

    else
    {
        ccw_dir_value = 0;
    }

    if (msg.ccw_speed > _max_speed)
    {
        ccw_speed = _max_speed;
    }
    
    correctSpeed(cw_speed);
    correctSpeed(ccw_speed);

    ROS_INFO_STREAM("corrected speed = " << cw_speed);
    //ROS_INFO_STREAM("direction pin = " << _dirPin);
    //ROS_INFO_STREAM("pwm pin = " << _pwmPin);

    digitalWrite(_cw_dirPin, cw_dir_value);
    pwmWrite(_cw_pwmPin, cw_speed);
    digitalWrite(_ccw_dirPin, ccw_dir_value);
    pwmWrite(_ccw_pwmPin, ccw_speed);
}

// void mySigintHandler(int sig)
// {


//     digitalWrite(26, 0);
//     pwmWrite(19, 0);
//     digitalWrite(21, 0);
//     pwmWrite(12, 0);

//     ros::shutdown();
// }


int main(int argc, char **argv)
{
    ros::init(argc, argv, "subscribe_to_speed");
    // ros::init(argc, argv, "subscribe_to_speed",ros::init_options::NoSigintHandler);
    // ros::NodeHandle nh;
    // signal(SIGINT, mySigintHandler);

    Motor motor;
    ros::spin();
}

    /*
    Motor motor1(19, 26);
    Motor motor2(12, 21);

    float i = 0;
    int j = 0;
    int k = 0;
    while(true)
    {

        //float speed = 0;

        j = (int)i;
        motor1.setSpeed(k);
        motor2.setSpeed(j); delay(5);

        i += 0.001;

        cout << "j= " << j << endl;
        cout << "k = " << k << endl;

    }
*/