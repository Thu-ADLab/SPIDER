from typing import Tuple
from spider.RL.reward.FstateTrajectoryReward import FstateTrajectoryReward
from spider.RL.reward.FstateControlReward import FstateControlReward
'''
qzl:
todo: reward function的写法需要大改动
可以仿照constraints checker, 写一个rewardcollection
要先预先定义几个形式的state和action，然后根据不同形式的state和action来写reward function

在reward下面加一个utils.py 计算各种常用reward的计算
'''

# 抽象类
# class BaseRewardFunction:
#     def __init__(self):
#
#         pass
#
#     def evaluate(self, state, action, next_state) -> Tuple[float, bool]:
#
#         if state is None :
#             return 0.0, False # todo: reward应该直接赋0，但是是否done其实要判断的，因为存在一开局就结局的情况
#
#         min_reward = -10
#
#         # 检查当前时刻是否碰撞
#         # 检查
#         reward = 0
        #
        # if np.any(np.array(traj.v) > self.config["max_speed"]):
        #     reward = min_reward
        #     break
        #
        # if np.any(np.array(traj.v) < self.config["min_speed"]):
        #     reward = min_reward
        #     break
        # if np.any(np.array(traj.a) > self.config["max_acceleration"]):
        #     reward = min_reward
        #     break
        # if np.any(np.array(traj.a) < -self.config["max_deceleration"]):
        #     reward = min_reward
        #     break
        # if np.any(np.abs(traj.curvature) > self.config["max_curvature"]):
        #     reward = min_reward
        #     break
        # if self.collision_checker.check_trajectory(traj, obstacles):
        #     reward = min_reward
        #     break