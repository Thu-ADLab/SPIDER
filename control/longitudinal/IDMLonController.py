import numpy as np
# import math

class IDMLonController(object):

    def __init__(self, a=1.0, b=2, delta=2, s0=8.0, T=2.5):
        """
        Initialize the IDM controller with the given parameters.

        a: maximum acceleration of the vehicle
        b: comfortable deceleration of the vehicle
        delta: acceleration exponent
        s0: minimum distance to the front vehicle
        T: safe time headway to the front vehicle
        """
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.T = T

    def run_step(self, v, v_desired, v_front, s):
        """
        Execute one step of longitudinal control.

        v: current speed of the vehicle
        v_desired: desired speed of the vehicle
        v_front: speed of the vehicle in front
        s: distance to the vehicle in front

        return: throttle control in the range [0, 1]
        """
        acc = self._calc_acceleration(v, v_desired, v_front, s)
        return acc#np.clip(acc, -self.a, self.a)

    def _calc_acceleration(self, v, v_desired, v_front, s):
        delta_v = v - v_front
        s_star = self.s0 + max(0, v * self.T + v * delta_v / (2 * np.sqrt(self.a * self.b)))
        return self.a * (1 - (v / v_desired) ** self.delta - (s_star / s) ** 2)

