#!/usr/bin/env python3
import time
import rospy
from motor_control.msg import MotorCommand

from gpiozero import Servo


class servo_turn:
    """
    Servo class definition, which subscribes to the topic /servo_angle to get angle commands for the servomotor
    """
    def __init__(self):
        """
        Servo class default constructor. Its attributes are:
        _servoPin: GPIO pin that sends PWM signals to the servo
        pwm:       object from the RPi.GPIO library that handles the servo properties and movement
        angle_sub: subscriber to the /servo_angle topic, with motor_control::MotorCommand message type and update_angle as callback function
        """
        self._servoPin = rospy.get_param("~servoPin")
        self.servo = Servo(self._servoPin)
        self.angle_sub = rospy.Subscriber("/motor_control", MotorCommand, self.update_angle)
    
    def update_angle(self,msg):
        """
        Function that converts the angle command to the appropriate duty cycle to
        handle the movement of the servo
        """
        if msg.angle==1:
            self.servo.max()
        
        if msg.angle==10:
            self.servo.min()

    #Return the rotors the the upward positions
    def return_pos_up(self):
        self.servo.max()

if __name__ == '__main__':
    rospy.init_node('subscribe_to_angle') # Node initialization #, anonymous=True)
    myServo = servo_turn()                     # Instantiation of the Servo class
    rospy.spin()
    rospy.on_shutdown(myServo.return_pos_up)


