import numpy as np
import math

class PurePursuitController(object):

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
        from spider.utils.geometry import resample_polyline

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
