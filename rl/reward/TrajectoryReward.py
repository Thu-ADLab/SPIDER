from typing import Tuple, Sequence
import numpy as np
import math
import torch


import spider
from spider.rl.reward.BaseReward import BaseReward

from spider.evaluator import CartCostEvaluator
from spider.constraints import CartConstriantChecker
from spider.utils.collision import BoxCollisionChecker



class TrajectoryReward(BaseReward):
    def __init__(self,
                 valid_x_range=None, valid_y_range=None,
                 des_x_range=None, des_y_range=None,
                 ego_size=(5., 2.),
                 ):
        super().__init__()

        self.valid_x_range = (-math.inf, math.inf) if valid_x_range is None else valid_x_range # (-10., 250.0)
        self.valid_y_range = (-math.inf, math.inf) if valid_y_range is None else valid_y_range # (-10.,10.)

        self.destination_x_range = (math.inf, math.inf) if des_x_range is None else des_x_range #(245., 255.0)
        self.destination_y_range = (math.inf, math.inf) if des_y_range is None else des_y_range #(-5., 5.)

        self.max_reward = 10.0
        self.finish_reward = 5.0
        self.punish_reward = -10.0

        # self.trajectory_evaluator = CartCostEvaluator(0.1, 100,1.0)

        _ego_length, _ego_width = ego_size
        self.collision_checker = BoxCollisionChecker(_ego_length, _ego_width)
        self.constraint_checker = CartConstriantChecker({}, None) # check kinematics only


    def evaluate_log(self, observation, plan, next_observation) -> Tuple[float, bool]:
        """
        Evaluate the reward and termination condition for the given log.

        :param observation: the initial observation
        :param plan: the plan that lead to the next observation
        :param next_observation: the observation we are evaluating the log for
        :return: a tuple of (reward, done)
        """
        ego = observation[0]
        ego_, perc_, lmap_ = next_observation
        ego_x_, ego_y_ = ego_.x(), ego_.y()

        range_reward, done = self._out_range_reward(ego_x_, ego_y_)
        if done:
            return range_reward, done

        c_reward, done = self._collision_reward(ego_, perc_)
        if done:
            return c_reward, done

        # _break_kinematics_reward, done = self._break_kinematics_reward(plan)
        d_reward, done = self._destination_reward(ego_x_, ego_y_)
        cline_reward, _ = self._centerline_reward(ego_, lmap_)
        val_reward = self._efficiency_reward(ego_)[0] + self._comfort_reward(ego,ego_)[0] + self._stuck_reward(ego_)[0]
        delay_reward, _ = self._delay_reward()

        reward = sum([range_reward, c_reward, d_reward, val_reward, delay_reward])
        reward = max([reward, self.punish_reward])
        reward = min([reward, self.max_reward])

        return reward, done

    def evaluate_exp(self, *args) -> Tuple[float, bool]:
        raise NotImplementedError("Not implemented. Ready for model-based reward.")


    def _delay_reward(self):
        return -0.5, False

    def _efficiency_reward(self, ego_):
        v = ego_.v()
        return (v ** 2)/20, False

    def _comfort_reward(self, ego, ego_):
        delta_v = ego_.v() - ego.v()
        return -abs(delta_v) , False

    def _stuck_reward(self, ego_):
        if hasattr(self, "_stuck_count"):
            self._stuck_count = 0

        v = ego_.v()
        if v < 0.1:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        done = self._stuck_count > 15

        return - self._stuck_count * 0.5, done


    # def _traj_eval_reward(self, traj):
    #     cost = self.trajectory_evaluator.evaluate(traj)
    #     return -cost / 1000 , False

    def _break_kinematics_reward(self, traj):
        if self.constraint_checker.check_kinematics(traj):
            return self.punish_reward, False
        else:
            return 0.0, False

    def _centerline_reward(self, ego_state, local_map):
        if local_map is None:
            return 0., False
        nearest_lane_id, dist = local_map.match_lane(ego_state, return_dist=True)
        return - (dist/1.75) * 3, False
        #     if distance > 0:
        #         direction = closest_lane.direction_at(distance)
        #         if direction == 1:
        #             if centerline.y(x) < y:
        #                 return 0.0, False
        #             else:
        #                 return self.punish_reward, True
        #         elif direction == -1:
        #             if centerline.y(x) > y:
        #                 return 0.0, False
        #             else:
        #                 return self.punish_reward, True
        #         else:
        #             return self.punish_reward, True
        #     else:
        #         return self.punish_reward, True
        # else:
        #     return self.punish_reward, True


    def _out_range_reward(self, ego_x, ego_y) -> Tuple[float, bool]:
        if not self.valid_x_range[0] <= ego_x <= self.valid_x_range[1]:
            return self.punish_reward, True
        elif not self.valid_y_range[0] <= ego_y <= self.valid_y_range[1]:
            return self.punish_reward, True
        else:
            return 0.0, False

    def _destination_reward(self, ego_x, ego_y) -> Tuple[float, bool]:
        if self.destination_x_range[0] <= ego_x <= self.destination_x_range[1]:
            if self.destination_y_range[0] <= ego_y <= self.destination_y_range[1]:
                return self.finish_reward, True
        return 0.0, False

    def _collision_reward(self, ego_state, perception):
        collision = self.collision_checker.check_state(ego_state, perception)
        if collision:
            return self.punish_reward, True
        else:
            return 0.0, False


    # def set_trajectory_candidates(self, trajectory_candidates: Sequence[FrenetTrajectory]) -> None:
    #     self.trajectory_candidates = trajectory_candidates
    #
    # def evaluate(self, state=None, action=None, next_state=None) -> Tuple[float, bool]:
    #     if state is None or action is None or next_state is None:
    #         return 0.0, False
    #
    #     # done与否是用碰撞检测来计算的
    #     next_state = next_state.view(self.config["state_veh_num"], self.config["state_feat_num"])
    #     collision_reward, collision = self.collision_reward(next_state)
    #     if collision:
    #         return collision_reward, True
    #
    #
    #     if next_state[0][1] > self.config["finishing_line"]:
    #         finish_reward = 10.
    #         done = True
    #     else:
    #         finish_reward = 0.
    #         done = False
    #     comfort_reward = self.comfort_reward(action)
    #     eff_reward = self.efficiency_reward(action)
    #     safety_reward = self.safety_reward(action)
    #     feasibility_reward = self.feasibility_reward(action)
    #     reward = max([(finish_reward + comfort_reward + eff_reward + safety_reward + feasibility_reward) / 1000,
    #                   self.punish_reward])
    #     return reward, done
    #
    #
    #
    # def collision_reward(self, next_state)  -> Tuple[float, bool]:
    #     # 障碍物collision
    #     # qzl:如果以轨迹为规划结果，碰撞的reward应该是下一时刻的next state是否碰撞还是这条轨迹上所有轨迹点是否碰撞？
    #     # 这里的车辆长宽没包括进来
    #
    #     # 如果是frenet坐标: (presence, s, l, s_dot, l_dot(l_prime), length, width)
    #     presence, s, l, s_dot, l_prime, length, width = next_state[0].tolist()
    #     ego_bbox = [s,l,self.config["ego_veh_length"], self.config["ego_veh_width"], math.atan(l_prime)]
    #     # qzl:因为默认没有负速度，所以不需要arctan2判断车头方向
    #     ego_vertices = obb2vertices(ego_bbox)
    #     for i, info in enumerate(next_state[1:]):
    #         presence, s, l, s_dot, l_prime, length, width = info.tolist()
    #         obs_bbox = [s, l, self.obstacle_length, self.obstacle_width, math.atan(l_prime)]
    #         obs_vertices = obb2vertices(obs_bbox)
    #         if SAT_check(ego_vertices, obs_vertices):
    #             # 撞了
    #             return self.punish_reward, True
    #
    #     return 0.0, False
    #
    # def comfort_reward(self, action) -> float:
    #     traj = self.trajectory_candidates[action]
    #     comfort = -np.sum(
    #         self.weight_long_comfort * np.array(traj.s_3dot) ** 2 + \
    #         self.weight_lat_comfort * np.array(traj.l_3prime) ** 2
    #     ).item()
    #     return comfort
    #
    #
    # def efficiency_reward(self, action) -> float:
    #     traj = self.trajectory_candidates[action]
    #     efficiency = (traj.s[-1] - traj.s[0]) ** 2 + \
    #                  5 * traj.s_dot[-1] ** 2
    #
    #     return efficiency
    #
    # def safety_reward(self, action) -> float:
    #     traj = self.trajectory_candidates[action]
    #     # 计算到每条车道中心线的距离并取最小
    #     distances = np.abs(np.array(traj.l)[:, np.newaxis] - np.array(self.centerline_ls))
    #     min_dist = np.min(distances, axis=1)
    #     safety = -np.sum(min_dist ** 2).item()
    #     return safety
    #
    # def feasibility_reward(self, action) -> float:
    #     traj = self.trajectory_candidates[action]
    #     if np.any(np.array(traj.v) > self.config["max_speed"]) or \
    #             np.any(np.array(traj.v) < self.config["min_speed"]) or \
    #             np.any(np.array(traj.a) > self.config["max_acceleration"]) or \
    #             np.any(np.array(traj.a) < -self.config["max_deceleration"]) or \
    #             np.any(np.abs(traj.curvature) > self.config["max_curvature"]) or \
    #             np.any(np.array(traj.l) < self.l_lower_bound) or \
    #             np.any(np.array(traj.l) > self.l_upper_bound):
    #         return self.punish_reward
    #     else:
    #         return 0
    #

# if __name__ == '__main__':

