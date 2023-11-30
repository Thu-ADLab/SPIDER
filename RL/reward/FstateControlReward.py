from typing import Tuple, Sequence
import numpy as np
import math
import torch

from spider.utils.collision.SAT import SAT_check
from spider.elements.Box import obb2vertices
from spider.elements.trajectory import FrenetTrajectory


class FstateControlReward:
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

    # def set_trajectory_candidates(self, trajectory_candidates: Sequence[FrenetTrajectory]) -> None:
    #     self.trajectory_candidates = trajectory_candidates

    def evaluate(self, state, action, next_state) -> Tuple[float, bool]:
        if state is None or action is None or next_state is None:
            return 0.0, False

        # done与否是用碰撞检测来计算的
        state = state.view(self.config["state_veh_num"], self.config["state_feat_num"])
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
        eff_reward = self.efficiency_reward(state, next_state)
        safety_reward = self.safety_reward(next_state)
        feasibility_reward = self.feasibility_reward(next_state, action)
        reward = max([(finish_reward + comfort_reward + eff_reward + safety_reward + feasibility_reward)/10,
                      self.punish_reward])

        # print('reward:', finish_reward ,comfort_reward , eff_reward , safety_reward , feasibility_reward)
        # print("sum:",reward)
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
        # qzl: 要不要normalize？
        comfort = -torch.sum(action**2).item()
        return comfort*0.1


    def efficiency_reward(self, state, next_state) -> float:
        s0 = state[0][1].item()
        s = next_state[0][1].item()
        s_dot = next_state[0][3].item()
        efficiency = (s-s0) ** 2 + s_dot ** 2
        return efficiency

    def safety_reward(self, next_state) -> float:
        l = next_state[0][2].item()
        l_prime = next_state[0][4].item()

        # 计算到每条车道中心线的距离并取最小
        distances = np.abs(l - np.array(self.centerline_ls))
        min_dist = np.min(distances).item()
        safety = - min_dist ** 2 - l_prime ** 2
        return safety

    def feasibility_reward(self, next_state, action) -> float:
        presence, s, l, s_dot, l_prime, length, width = next_state[0].tolist()
        acc, steer = action.tolist()
        # traj = self.trajectory_candidates[action]
        if s_dot > self.config["max_speed"] or \
                s_dot < self.config["min_speed"] or \
                acc > self.config["max_acceleration"] or \
                acc < -self.config["max_deceleration"] or \
                steer > self.config["max_steer"] or \
                steer < -self.config["max_steer"] or \
                l < self.l_lower_bound or \
                l > self.l_upper_bound:
            return self.punish_reward
        else:
            return 0

