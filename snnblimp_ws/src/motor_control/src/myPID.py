#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class PID:
    
    def __init__(self, P, I, D, dt, simple):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        
        self.dt = dt
        
        self.heights = 0
        
        if simple:
            self.clear_simple()
        else:
            self.clear()
    
    def clear_simple(self):
        """
        Variables initialization for update_simple()
        
        Parameters
        ----------
        self.integral : Integral term
        self.previous_error : Error at the previous timestep
        self.window_up : Saturation value for the integral term
        """
        self.integral                       = 0
        self.previous_error                 = 0
        self.previous_derivative            = 0
        self.previous_previous_derivative   = 0
        self.window_up      = 20
        self.meas_prev      = 0      

    
    def update_simple(self, error):
        """
        PID implementation based on https://en.wikipedia.org/wiki/PID_controller#Pseudocode
        
        previous_error := 0
        integral := 0
        
        loop:
            error := setpoint − measured_value
            integral := integral + error × dt
            derivative := (error − previous_error) / dt
            output := Kp × error + Ki × integral + Kd × derivative
            previous_error := error
            wait(dt)
            goto loop
        
        """
        
        self.integral += error * self.dt
        
        # Saturate integral
        if self.integral > self.window_up:
            self.integral = self.window_up
        elif self.integral < -self.window_up:
            self.integral = -self.window_up
        
        self.derivative = (error - self.previous_error)/self.dt
        
        self.previous_error = error
        
        # u = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative
        
        return self.Kp * error, self.Ki * self.integral, self.Kd * self.derivative
    
    def update_simple_3m(self, error):
        """
        PID implementation based on https://en.wikipedia.org/wiki/PID_controller#Pseudocode
        derivative based on : https://web.media.mit.edu/~crtaylor/calculator.html
        
        previous_error := 0
        integral := 0
        
        loop:
            error := setpoint − measured_value
            integral := integral + error × dt
            derivative := (3*e(t) -4*e(t-1) + e(t-2))/ 2*dt
            output := Kp × error + Ki × integral + Kd × derivative
            previous_error := error
            wait(dt)
            goto loop
        
        """
        
        self.integral += error * self.dt
        
        # Saturate integral
        if self.integral > self.window_up:
            self.integral = self.window_up
        elif self.integral < -self.window_up:
            self.integral = -self.window_up
        
        self.derivative_current = (error - self.previous_error)/self.dt

        self.derivative = (self.derivative_current+self.previous_derivative)/2

        self.previous_error = error
        self.previous_derivative = self.derivative_current

        
        return self.Kp * error, self.Ki * self.integral, self.Kd * self.derivative
    
    
    def update_simple_4m(self, error):
        """
        PID implementation based on https://en.wikipedia.org/wiki/PID_controller#Pseudocode
        derivative based on : https://web.media.mit.edu/~crtaylor/calculator.html
        
        previous_error := 0
        integral := 0
        
        loop:
            error := setpoint − measured_value
            integral := integral + error × dt
            derivative := (3*e(t) -4*e(t-1) + e(t-2))/ 2*dt
            output := Kp × error + Ki × integral + Kd × derivative
            previous_error := error
            wait(dt)
            goto loop
        
        """
        
        self.integral += error * self.dt
        
        # Saturate integral
        if self.integral > self.window_up:
            self.integral = self.window_up
        elif self.integral < -self.window_up:
            self.integral = -self.window_up
        
        self.current_derivative = (error - self.previous_error)/self.dt

        self.derivative = (self.current_derivative+self.previous_derivative + self.previous_previous_derivative)/3

        self.previous_error = error
        self.previous_previous_derivative = self.previous_derivative
        self.previous_derivative = self.current_derivative

        
        return self.Kp * error, self.Ki * self.integral, self.Kd * self.derivative
    
    
    
    def update_simple_h(self, error, meas):
        """
        PID implementation based on https://en.wikipedia.org/wiki/PID_controller#Pseudocode
        derivative based on : https://web.media.mit.edu/~crtaylor/calculator.html
        
        previous_error := 0
        integral := 0
        
        loop:
            error := setpoint − measured_value
            integral := integral + error × dt
            derivative := (3*e(t) -4*e(t-1) + e(t-2))/ 2*dt
            output := Kp × error + Ki × integral + Kd × derivative
            previous_error := error
            wait(dt)
            goto loop
        
        """
        
        self.integral += error * self.dt
        
        # Saturate integral
        if self.integral > self.window_up:
            self.integral = self.window_up
        elif self.integral < -self.window_up:
            self.integral = -self.window_up
        
        self.derivative = -(meas - self.meas_prev)/self.dt

        self.meas_prev = meas
        
        # u = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative
        
        return self.Kp * error, self.Ki * self.integral, self.Kd * self.derivative
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    