import warnings

import math
from collections import deque
import numpy as np

import spider.elements as elm
from spider.utils.geometry import resample_polyline


class SimpleController(object):

    def __init__(self):
        self._lon_controller = LonController()
        self._lat_controller = PurePuesuitController()

        self._max_brake = 0.5
        self._max_throt = 0.75
        self._max_steer = 0.8

    def get_control(self, trajectory:elm.Trajectory, ego_veh_state:elm.VehicleState=None):
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


class LonController(object):

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


class PurePuesuitController(object):

    def __init__(self):
        pass

    def run_step(self, trajectory_array, ego_pose, current_speed):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        """

        ego_loc = ego_pose[:2]
        control_point = self._control_point(trajectory_array, ego_loc, current_speed)

        if len(control_point) < 2:
            return 0.0
        return self._purepersuit_control(control_point, ego_pose)

    def _control_point(self, trajectory_arr, ego_loc, current_speed, resolution=0.1):

        if current_speed > 10:
            control_target_dt = 0.5 - (current_speed - 10) * 0.01
        else:
            control_target_dt = 0.5
        # control_target_dt = 0.4

        control_target_distance = control_target_dt * current_speed  ## m
        if control_target_distance < 3:
            control_target_distance = 3

        trajectory_dense = resample_polyline(trajectory_arr, resolution)

        end_idx = self.get_next_idx(ego_loc, trajectory_dense, control_target_distance)
        wp_loc = trajectory_dense[end_idx]

        return wp_loc

    def _purepersuit_control(self, waypoint, ego_pose):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """

        ego_x, ego_y, ego_yaw = ego_pose
        # ego_yaw = ego_vehicle.yaw
        # ego_x = ego_vehicle.x
        # ego_y = ego_vehicle.y

        v_vec = np.array([math.cos(ego_yaw),
                          math.sin(ego_yaw),
                          0.0])

        target_x = waypoint[0]
        target_y = waypoint[1]

        w_vec = np.array([target_x -
                          ego_x, target_y -
                          ego_y, 0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        lf = 1.2
        lr = 1.95
        lwb = lf + lr

        v_rear_x = ego_x - v_vec[0] * lr / np.linalg.norm(v_vec)
        v_rear_y = ego_y - v_vec[1] * lr / np.linalg.norm(v_vec)
        l = (target_x - v_rear_x) * (target_x - v_rear_x) + (target_y - v_rear_y) * (target_y - v_rear_y)
        l = math.sqrt(l)

        theta = np.arctan(2 * np.sin(_dot) * lwb / l)

        k = 1.0  # XXX: np.pi/180*50
        theta = theta * k
        return theta

    # def convert_trajectory_to_ndarray(self, trajectory):
    #     trajectory_array = [(pose.pose.position.x, pose.pose.position.y) for pose in trajectory.poses]
    #     return np.array(trajectory_array)

    def get_idx(self, loc, trajectory):
        dist = np.linalg.norm(trajectory - loc, axis=1)
        idx = np.argmin(dist)
        return idx

    def get_next_idx(self, start_loc, trajectory, distance):

        start_idx = self.get_idx(start_loc, trajectory)
        dist_list = np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        for end_idx in range(start_idx, len(trajectory) - 1):
            if dist_list[end_idx] > dist_list[start_idx] + distance:
                return end_idx
