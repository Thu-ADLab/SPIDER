from typing import Tuple, Sequence
import numpy as np
import math
import torch

from spider.utils.collision.SAT import SAT_check
from spider.elements.Box import obb2vertices
from spider.elements.trajectory import FrenetTrajectory


class FstateTrajectoryReward:
    def __init__(self, config: dict):
        self.trajectory_candidates: Sequence[FrenetTrajectory] = []
        self.config = config

        self.weight_lat_comfort = 5.0
        self.weight_long_comfort = 1.0

        self.weight_comfort = 1.0
        self.weight_efficiency = 1.0
        self.weight_safety = 1.0

        self.max_reward = 10.0
        self.punish_reward = -10.0

        self.centerline_ls = self.config["end_l_candidates"]
        self.lane_width = self.config["lane_width"]
        self.l_lower_bound = min(self.centerline_ls) - 0.5 * self.lane_width + 0.5 * self.config["ego_veh_width"]
        self.l_upper_bound = max(self.centerline_ls) + 0.5 * self.lane_width - 0.5 * self.config["ego_veh_width"]
        self.obstacle_length = 5.0
        self.obstacle_width = 2.0

    def set_trajectory_candidates(self, trajectory_candidates: Sequence[FrenetTrajectory]) -> None:
        self.trajectory_candidates = trajectory_candidates

    def evaluate(self, state, action, next_state) -> Tuple[float, bool]:
        if state is None or action is None or next_state is None:
            return 0.0, False

        # done与否是用碰撞检测来计算的
        next_state = next_state.view(self.config["state_veh_num"], self.config["state_feat_num"])
        collision_reward, collision = self.collision_reward(next_state)
        if collision:
            return collision_reward, True


        if next_state[0][1] > self.config["finishing_line"]:
            finish_reward = 10.
            done = True
        else:
            finish_reward = 0.
            done = False
        comfort_reward = self.comfort_reward(action)
        eff_reward = self.efficiency_reward(action)
        safety_reward = self.safety_reward(action)
        feasibility_reward = self.feasibility_reward(action)
        reward = max([(finish_reward + comfort_reward + eff_reward + safety_reward + feasibility_reward) / 1000,
                      self.punish_reward])
        return reward, done



    def collision_reward(self, next_state)  -> Tuple[float, bool]:
        # 障碍物collision
        # qzl:todo: 如果以轨迹为规划结果，碰撞的reward应该是下一时刻的next state是否碰撞还是这条轨迹上所有轨迹点是否碰撞？
        # todo: 这里的车辆长宽没包括进来

        # 如果是frenet坐标: (presence, s, l, s_dot, l_dot(l_prime), length, width)
        presence, s, l, s_dot, l_prime, length, width = next_state[0].tolist()
        ego_bbox = [s,l,self.config["ego_veh_length"], self.config["ego_veh_width"], math.atan(l_prime)]
        # qzl:因为默认没有负速度，所以不需要arctan2判断车头方向
        ego_vertices = obb2vertices(ego_bbox)
        for i, info in enumerate(next_state[1:]):
            presence, s, l, s_dot, l_prime, length, width = info.tolist()
            obs_bbox = [s, l, self.obstacle_length, self.obstacle_width, math.atan(l_prime)]
            obs_vertices = obb2vertices(obs_bbox)
            if SAT_check(ego_vertices, obs_vertices):
                # 撞了
                return self.punish_reward, True

        return 0.0, False

    def comfort_reward(self, action) -> float:
        traj = self.trajectory_candidates[action]
        comfort = -np.sum(
            self.weight_long_comfort * np.array(traj.s_3dot) ** 2 + \
            self.weight_lat_comfort * np.array(traj.l_3prime) ** 2
        ).item() # todo：l_3dot？
        return comfort


    def efficiency_reward(self, action) -> float:
        traj = self.trajectory_candidates[action]
        efficiency = (traj.s[-1] - traj.s[0]) ** 2 + \
                     5 * traj.s_dot[-1] ** 2

        return efficiency

    def safety_reward(self, action) -> float:
        traj = self.trajectory_candidates[action]
        # 计算到每条车道中心线的距离并取最小
        distances = np.abs(np.array(traj.l)[:, np.newaxis] - np.array(self.centerline_ls))
        min_dist = np.min(distances, axis=1)
        safety = -np.sum(min_dist ** 2).item()
        return safety

    def feasibility_reward(self, action) -> float:
        traj = self.trajectory_candidates[action]
        if np.any(np.array(traj.v) > self.config["max_speed"]) or \
                np.any(np.array(traj.v) < self.config["min_speed"]) or \
                np.any(np.array(traj.a) > self.config["max_acceleration"]) or \
                np.any(np.array(traj.a) < -self.config["max_deceleration"]) or \
                np.any(np.abs(traj.curvature) > self.config["max_curvature"]) or \
                np.any(np.array(traj.l) < self.l_lower_bound) or \
                np.any(np.array(traj.l) > self.l_upper_bound):
            return self.punish_reward
        else:
            return 0

