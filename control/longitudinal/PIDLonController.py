from collections import deque
import numpy as np

class PIDLonController(object):

    def __init__(self):
        """
        vehicle: actor to apply to local planner logic onto
        K_P: Proportional term
        K_D: Differential term
        K_I: Integral term
        dt: time differential in seconds
        """
        self._K_P = 0.25 / 3.6
        self._K_D = 0  # .01
        self._K_I = 0  # 0.012 #FIXME: To stop accmulate error when repectly require control signal in the same state.
        self._dt = 0.1  # TODO: timestep
        self._integ = 0.0
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

        target_speed: target speed in Km/h
        return: throttle control in the range [0, 1]
        """

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [-1, 1]
        """
        if target_speed == 0:
            return -1

        target_speed = target_speed * 3.6
        current_speed = current_speed * 3.6

        _e = (target_speed - current_speed)
        self._integ += _e * self._dt
        self._e_buffer.append(_e)

        if current_speed < 2:
            self._integ = 0

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = self._integ

        else:
            _de = 0.0
            _ie = 0.0
        kp = self._K_P
        ki = self._K_I
        kd = self._K_D

        if target_speed < 5:
            ki = 0
            kd = 0

        calculate_value = np.clip((kp * _e) + (kd * _de) + (ki * _ie), -1.0, 1.0)
        return calculate_value

