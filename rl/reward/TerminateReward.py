from typing import Tuple, Sequence
import numpy as np
import math
import torch


import spider
from spider.rl.reward.BaseReward import BaseReward

from spider.evaluator import CartCostEvaluator
from spider.constraints import CartConstriantChecker
from spider.utils.collision import BoxCollisionChecker



class TerminateReward(BaseReward):
    '''
    仅检查done
    '''
    def __init__(self,
                 valid_x_range=None, valid_y_range=None,
                 des_x_range=None, des_y_range=None,
                 collision_done=False, ego_size=(5.,2.),
                 kinematics_done=False, constraints={}
                 ):
        super().__init__()

        self.valid_x_range = (-math.inf, math.inf) if valid_x_range is None else valid_x_range # (-10., 250.0)
        self.valid_y_range = (-math.inf, math.inf) if valid_y_range is None else valid_y_range # (-10.,10.)

        self.destination_x_range = (math.inf, math.inf) if des_x_range is None else des_x_range #(245., 255.0)
        self.destination_y_range = (math.inf, math.inf) if des_y_range is None else des_y_range #(-5., 5.)

        # self.max_reward = 10.0
        self.finish_reward = 10.0
        self.punish_reward = -10.0

        self.collision_done = collision_done
        if collision_done:
            self.collision_checker = BoxCollisionChecker(*ego_size)

        if kinematics_done: # todo: 没有启用
            self.constraint_checker = CartConstriantChecker({}, None) # check kinematics only


    def evaluate_log(self, observation, plan, next_observation) -> Tuple[float, bool]:
        """
        Evaluate the reward and termination condition for the given log.

        :param observation: the initial observation
        :param plan: the plan that lead to the next observation
        :param next_observation: the observation we are evaluating the log for
        :return: a tuple of (reward, done)
        """
        ego_, perc_, lmap_ = next_observation
        ego_x_, ego_y_ = ego_.x(), ego_.y()

        range_reward, done = self._out_range_reward(ego_x_, ego_y_)
        if done:
            return range_reward, done

        if self.collision_done:
            c_reward, done = self._collision_reward(ego_, perc_)
            if done:
                return c_reward, done

        # _break_kinematics_reward, done = self._break_kinematics_reward(plan)
        d_reward, done = self._destination_reward(ego_x_, ego_y_)
        delay_reward, _ = self._delay_reward()

        return d_reward+delay_reward, done

    def evaluate_exp(self, *args) -> Tuple[float, bool]:
        raise NotImplementedError("Not implemented. Ready for model-based reward.")

    def _delay_reward(self):
        return -0.5, False


    def _break_kinematics_reward(self, traj):
        if self.constraint_checker.check_kinematics(traj):
            return self.punish_reward, False
        else:
            return 0.0, False


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