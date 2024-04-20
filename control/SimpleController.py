import warnings

import numpy as np

from typing import TypeVar
from spider.control.lateral import PurePursuitController
from spider.control.longitudinal import PIDLonController

Trajectory = TypeVar("Trajectory") # 避免循环引用
VehicleState = TypeVar("VehicleState")

class SimpleController(object):

    def __init__(self):
        self._lon_controller = PIDLonController()
        self._lat_controller = PurePursuitController()

        self._max_brake = 0.5
        self._max_throt = 0.75
        self._max_steer = 0.8

    def get_control(self, trajectory:Trajectory, ego_veh_state:VehicleState=None):
        # target
        target_speed = trajectory.v[-1]
        traj_arr = trajectory.trajectory_array

        # current
        if ego_veh_state is None:
            warnings.warn("Ego_state is not provided. Using the trajectory starting point instead.")
            current_speed = trajectory.v[0]
            ego_pose = np.array([trajectory.x[0], trajectory.y[0], trajectory.heading[0]])
        else:
            current_speed = ego_veh_state.v()
            ego_pose = np.array([ego_veh_state.x(), ego_veh_state.y(), ego_veh_state.yaw()])

        acc, steering = self.get_control_(traj_arr, target_speed, ego_pose, current_speed)

        return acc, steering

    def get_control_(self, trajectory_array:np.ndarray, target_speed, current_pose, current_speed):
        '''
        :param trajectory_array: [[x1,y1],[x2,y2]]
        :param target_speed:
        :param current_pose: [x,y,yaw]
        :param current_speed:
        :return:
        '''
        acc = self._lon_controller.run_step(target_speed, current_speed)
        steering = self._lat_controller.run_step(np.asarray(trajectory_array), current_pose, current_speed)

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

if __name__ == '__main__':
    ctl = SimpleController()
    a, st =  ctl.get_control_(np.array([[0,0],[1,1]]), 10, [0,0,0], 10)
    print(a, st)

