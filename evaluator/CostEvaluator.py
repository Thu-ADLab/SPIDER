from spider.elements.trajectory import Trajectory, FrenetTrajectory
import numpy as np
from typing import List


class CartCostEvaluator:
    def __init__(self):
        # self.weight_lat_comfort = 5.0
        # self.weight_long_comfort = 1.0

        self.weight_comfort = 1.0
        self.weight_efficiency = 40.0
        self.weight_safety = 1.0

    def evaluate(self, traj: FrenetTrajectory):
        '''
        评价舒适性、通行效率、安全性
        '''

        comfort = np.sum(np.asarray(traj.jerk) ** 2) + \
                    np.sum(np.abs(traj.centripetal_acceleration))

        efficiency = -np.sum(traj.v)

        safety = 0#np.sum(np.array(traj.l) ** 2)

        cost = self.weight_comfort * comfort +\
               self.weight_efficiency * efficiency +\
               self.weight_safety * safety

        return cost

    def evaluate_candidates(self, trajectory_list:List[FrenetTrajectory]):
        all_cost = [self.evaluate(t) for t in trajectory_list]
        idx = list(range(len(all_cost)))
        sorted_cost, sorted_idx = zip(*sorted(zip(all_cost, idx)))
        sorted_trajectories = [trajectory_list[i] for i in sorted_idx]
        # sorted_cost, sorted_trajectories = zip(*sorted(zip(all_cost, trajectory_list)))
        return sorted_trajectories,sorted_cost


class FrenetCostEvaluator:
    def __init__(self):
        self.weight_lat_comfort = 5.0
        self.weight_long_comfort = 1.0

        self.weight_comfort = 1.0
        self.weight_efficiency = 1.0
        self.weight_safety = 1.0

    def evaluate(self, traj: FrenetTrajectory):
        '''
        评价舒适性、通行效率、安全性
        todo:参数搞成可调的，加入更多可选的损失项，比如向心加速度什么的
        todo:横向的l_3dot和l_3prime始终难以统一，如何是好？
        '''

        comfort = np.sum(
            self.weight_long_comfort * np.array(traj.s_3dot) ** 2 + \
            self.weight_lat_comfort * np.array(traj.l_3prime) ** 2
            # self.weight_lat_comfort * np.array(traj.l_3dot) ** 2
        )

        efficiency = -(traj.s[-1]-traj.s[0]) ** 2 -\
                     5 * traj.s_dot[-1] ** 2

        safety = np.sum(np.array(traj.l) ** 2)
        if traj.l[-1] * traj.l[0] <0: # 异号，说明从左走到右了，惩罚
            safety *= 5

        cost = self.weight_comfort * comfort +\
               self.weight_efficiency * efficiency +\
               self.weight_safety * safety
        # cost = self.weight_efficiency * efficiency

        return cost

    def evaluate_candidates(self, trajectory_list:List[FrenetTrajectory]):
        all_cost = [self.evaluate(t) for t in trajectory_list]
        idx = list(range(len(all_cost)))
        sorted_cost, sorted_idx = zip(*sorted(zip(all_cost, idx)))
        sorted_trajectories = [trajectory_list[i] for i in sorted_idx]
        # sorted_cost, sorted_trajectories = zip(*sorted(zip(all_cost, trajectory_list)))
        return sorted_trajectories,sorted_cost


