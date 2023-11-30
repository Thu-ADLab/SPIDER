import numpy as np
import math
from typing import List


# class BicycleState(np.ndarray):
#     def __init__(self, shape, dtype=None, buffer=None, offset=0, strides=None, order=None):
#         super(BicycleState, self).__init__(shape, dtype, buffer, offset, strides, order)


class Bicycle:
    def __init__(self,
                 x=0., y=0., v=0., a=0., heading=0., steer=0., steer_velocity=0., *,
                 dt=0.1, wheelbase=3.,):
        # self.state = []
        # self.control = []
        self.setKinematics(x, y, v, a, heading, steer, steer_velocity)
        self.dt = dt
        self.wheelbase = wheelbase

    def setKinematics(self, x, y, v, a, heading, steer, steer_velocity=0.):
        self.x = x
        self.y = y
        self.heading = heading
        self.steer = steer
        self.velocity = v
        self.acceleration = a
        self.steer_velocity = steer_velocity

    def step(self, a, steer=None, steer_velocity=None, dt=0.):
        if dt==0.:
            dt = self.dt

        if steer is not None:
            v = self.velocity + a * dt
            heading = self.heading + dt * self.velocity * math.tan(steer) / self.wheelbase
            x = self.x + self.velocity * math.cos(heading) * dt
            y = self.y + self.velocity * math.sin(heading) * dt
            steer_velocity = (steer-self.steer) / dt
            self.setKinematics(x, y, v, a, heading, steer, steer_velocity)
        elif steer_velocity is not None:
            pass #TODO:补充
        else:
            raise ValueError("Invalid input")

    def derivative(self, x, y, dt=0.):
        if dt==0.:
            dt = self.dt
        dx = x - self.x
        dy = y - self.y
        heading = math.atan2(dy,dx)  # t+1时刻的
        v = dx / dt / math.cos(heading)  # t+1时刻的
        dv = v - self.velocity
        a = dv/dt
        dheading = heading-self.heading
        steer = math.atan(dheading * self.wheelbase / dt / self.velocity)
        steer_velocity = (steer - self.steer)/dt
        self.setKinematics(x, y, v, a, heading, steer, steer_velocity)

    # @staticmethod
    def accsteer2state(self, acc:np.ndarray, steer:np.ndarray):
        assert acc.shape[0] == steer.shape[0]
        #acc: 0-N-1, steer:0-N-1
        N = acc.shape[0]
        accumulation = np.zeros((N+1,N))
        accumulation[1:, :] = np.tril(np.ones((N, N)))
        V = accumulation @ acc * self.dt + self.velocity # 0-N
        heading = accumulation @ (V[:-1] * np.tan(steer) * self.dt / self.wheelbase) + self.heading# 0-N
        Y = accumulation @ (V[:-1] * np.sin(heading[:-1]) * self.dt) + self.y # 0-N
        X = accumulation @ (V[:-1] * np.cos(heading[:-1]) * self.dt) + self.x # 0-N
        return X, Y, V, heading



