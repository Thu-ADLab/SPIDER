#!/usr/bin/_env python
import math
from math import atan2, cos, sin

import torch
from numpy import clip, cos, sin, tan

# from Agent.world_model.agent_model.KinematicBicycleModel.libs.normalise_angle import normalise_angle
normalise_angle = lambda angle : atan2(sin(angle), cos(angle))


class Car:

    def __init__(self, init_x, init_y, init_yaw, dt):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = 0.0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(60)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        # Description parameters
        self.overall_length = 4.97
        self.overall_width = 1.964
        self.tyre_diameter = 0.4826
        self.tyre_width = 0.2032
        self.axle_track = 1.662
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'

        self.kbm = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

    def drive(self, throttle, steer):
        
        throttle = rand.uniform(150, 200)
        self.delta = steer
        self.x, self.y, self.yaw, self.v, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta)


class KinematicBicycleModel():

    def __init__(self, wheelbase=3.0, max_steer=1.5, dt=0.1, c_r=0.0, c_a=0.0):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
        :param c_r:             (float) vehicle's coefficient of resistance 
        :param c_a:             (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.dt_discre = 1 # Larger discre: More precies but slowly
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a

    def kinematic_model(self, x, y, yaw, velocity, throttle, delta):
        # Compute the local velocity in the x-axis
        for i in range(self.dt_discre):
            
            f_load = velocity * (self.c_r + self.c_a * velocity)

            velocity += (self.dt/self.dt_discre) * (throttle - f_load)
            if velocity <= 0:
                velocity = 0
            
            # Compute the radius and angular velocity of the kinematic bicycle model
            delta = clip(delta, -self.max_steer, self.max_steer)
            # Compute the state change rate
            x_dot = velocity * cos(yaw)
            y_dot = velocity * sin(yaw)
            omega = velocity * tan(delta) / self.wheelbase

            # Compute the final state using the discrete time model
            x += x_dot * (self.dt/self.dt_discre)
            y += y_dot * (self.dt/self.dt_discre)
            yaw += omega * (self.dt/self.dt_discre)
            yaw = normalise_angle(yaw)
        return x, y, yaw, velocity, delta, omega
    
    def calculate_a_from_data(self, x, y, yaw, velocity, x2, y2, yaw2, velocity2):
        f_load = velocity * (self.c_r + self.c_a * velocity)

        # velocity2 = ((x2-x)/cos(yaw)/self.dt + (y2-y)/sin(yaw)/self.dt)/2
        throttle = (velocity2 - velocity) / self.dt + f_load
        if velocity == 0:
            delta = 0
        else:
            delta = math.atan((yaw2 - yaw) / self.dt * self.wheelbase /velocity)

        return throttle, delta
    
class KinematicBicycleModel_Pytorch():

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.1, c_r=0.0, c_a=0.0):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
        :param c_r:             (float) vehicle's coefficient of resistance 
        :param c_a:             (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a
        self.dt_discre = 100

    def kinematic_model(self, x, y, yaw, velocity, throttle, delta):
        # Compute the local velocity in the x-axis
        for i in range(self.dt_discre):
            throttle = torch.mul(throttle, 1) # throttle * 10, steer (0-1)
            delta = torch.mul(delta, 1) # throttle * 10, steer (0-1)
            ca = torch.mul(velocity, self.c_a)
            temp = torch.add(ca, self.c_r)
            f_load = torch.mul(velocity, temp) # 
                                                                                                                
            dv = torch.mul(torch.sub(throttle, f_load), self.dt/self.dt_discre)
            velocity = torch.add(velocity, dv)  

            # Compute the state change rate
            x_dot = torch.mul(velocity, torch.cos(yaw))
            y_dot = torch.mul(velocity, torch.sin(yaw))
            omega = torch.mul(velocity, torch.tan(delta))
            omega = torch.mul(omega, 1/self.wheelbase)
            
            # Compute the final state using the discrete time model
            x = torch.add(x, torch.mul(x_dot, self.dt/self.dt_discre))
            y = torch.add(y, torch.mul(y_dot, self.dt/self.dt_discre))
            yaw = torch.add(yaw, torch.mul(omega, self.dt/self.dt_discre))
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

        return x, y, yaw, velocity, delta, omega
    
    def calculate_a_from_data(self, x, y, yaw, velocity, x2, y2, yaw2, velocity2):

        ca = torch.mul(velocity, self.c_a)
        temp = torch.add(ca, self.c_r)
        f_load = torch.mul(velocity, temp)         

        dv = torch.sub(velocity2, velocity)
        dv_dt = torch.div(dv, self.dt)
        throttle = torch.add(dv_dt, f_load)
        
        if velocity == 0:
            delta = torch.zeros_like(x)
        else:
            dyaw = torch.sub(yaw2, yaw)
            delta = torch.div(dyaw, self.dt/self.wheelbase)
            delta = torch.div(delta, velocity)
            delta = torch.atan(delta)
       
        return throttle, delta


def main():

    print("This script is not meant to be executable, and should be used as a library.")
    x1 = 0
    y1 = 0
    yaw1 = -1.6
    velocity1 = 1.931
    x2 = -0.04
    y2 = -0.23
    yaw2 = -1.62
    velocity2 = 2.533
    model = KinematicBicycleModel()
    
    throttle, delta = model.calculate_a_from_data(x1, y1, yaw1, velocity1, x2, y2, yaw2, velocity2)
    print("throttle, delta",throttle, delta)
    print("dx,dy",(x2-x1)/cos(yaw1),(y2-y1)/sin(yaw1))
    x, y, yaw, v, _, _ = model.kinematic_model(x1, y1, yaw1, velocity1, throttle, delta)
    print("x, y, yaw, v",x, y, yaw, v)

if __name__ == "__main__":
    main()
