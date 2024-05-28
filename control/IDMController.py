import warnings

import numpy as np

from typing import TypeVar
from spider.control.lateral import PurePursuitController
from spider.control.longitudinal import IDMLonController

Trajectory = TypeVar("Trajectory") # 避免循环引用
VehicleState = TypeVar("VehicleState")

class IDMController(object):

    def __init__(self):
        self._max_brake = 2.0
        self._max_throt = 1.0
        self._max_steer = 0.8

        self._lon_controller = IDMLonController(self._max_throt)
        self._lat_controller = PurePursuitController()



    def get_control(self, reference_line:np.ndarray, front_veh_speed, front_veh_dist, desired_speed,
                    current_pose, current_speed):
        acc = self._lon_controller.run_step(current_speed, desired_speed, front_veh_speed, front_veh_dist)
        steering = self._lat_controller.run_step(np.asarray(reference_line), current_pose, current_speed)

        acc = np.clip(acc, -self._max_brake, self._max_throt).item()
        steering = np.clip(steering, -self._max_steer, self._max_steer).item()
        return acc, steering


    def get_fallback_control(self, brake=None):
        '''
        Returns a control that will bring the vehicle to a halt in case of
        failure of control module or planning module.
        :param brake: amount of brake to apply
        :return: control
        '''
        acc = -self._max_brake if brake is None else -brake
        steering = 0.0

        return acc, steering


