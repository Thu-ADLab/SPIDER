import math
from typing import Union

import numpy as np
import torch.nn as nn
import torch


import spider.elements as elm
import spider.utils.lane_decision
from spider.sampler.LatticeSampler import LatticeSampler
from spider.data.DataBuffer import ExperienceBuffer

from spider.rl.state.StateConverter import KineStateEncoder
from spider.rl.action.ActionConverter import DiscreteTrajActionDecoder, DiscreteTrajActionEncoder
from spider.rl.policy.PPOPolicy import DiscretePPOPolicy
from spider.planner_zoo.BaseNeuralPlanner import BaseNeuralPlanner
from spider.utils.lane_decision import ConstLaneDecision


class MLP_Actor(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(  # 5-layer MLP
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_num)
        )

    def forward(self, x):
        return torch.nn.functional.softmax(self.mlp(x), dim=-1) # 返回的是每个离散动作的概率，所以做softmax


class MLP_Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=16):
        super().__init__()
        self.mlp = nn.Sequential(  # 5-layer MLP
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.mlp(x) # 返回的是对于状态的价值估计


class DiscretePPOPlanner(BaseNeuralPlanner):
    def __init__(self, config=None):
        super().__init__(config)

        self.state_encoder = KineStateEncoder(
            normalize=self.config["normalize"],
            relative=self.config["relative"],
            num_object=self.config["num_object"],
            x_range=self.config["longitudinal_range"],
            y_range=self.config["lateral_range"])


        self.trajectory_sampler = LatticeSampler(
            self.steps, self.dt,
            self.config["end_T_candidates"], self.config["end_v_candidates"],
            self.config["end_s_candidates"], self.config["end_l_candidates"],
            lane_decision=ConstLaneDecision(1), # todo:这个以后要改过来, 现在太笨了
            calc_by_need=True
        )

        self.action_decoder = DiscreteTrajActionDecoder(sampler=self.trajectory_sampler)
        # self.action_encoder = DiscreteTrajActionEncoder(sampler=self.trajectory_sampler)

        self.policy = DiscretePPOPolicy(
            MLP_Actor(self.state_encoder.state_dim, self.action_decoder.action_dim),
            MLP_Critic(self.state_encoder.state_dim),
            self.action_decoder.action_dim,
            enable_tensorboard=self.config["enable_tensorboard"],
            tensorboard_root=self.config["tensorboard_root"]
        )

        # self.reward = None
        #
        self.exp_buffer = ExperienceBuffer(maxlen=self.config["exp_buffer_maxlen"], forward_only=False, autosave=False)
        # self.exp_buffer.apply_to(self.policy, self.reward)


    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "steps": 20,
            "dt": 0.2,

            ####### 观测空间到状态空间变换 参数 #########
            "num_object": 5,
            "normalize": False,
            "relative": False,
            "longitudinal_range": (-50, 100),
            "lateral_range": (-20,20),

            ####### 离散动作空间 采样器参数 #########
            "end_s_candidates": (20,),
            "end_l_candidates": (-3.5, 0, 3.5),
            "end_v_candidates": (0.,60/3.6),#tuple(i * 60 / 3.6 / 2 for i in range(3)),
            "end_T_candidates": (4,),

            ####### 经验回放池 参数 #########
            "exp_buffer_maxlen": 100000,

            ####### 训练 参数 #########
            "batch_size": 64,

            # "learning_rate": 0.0001,
            "enable_tensorboard": False,
            "tensorboard_root": './tensorboard/'
            # "epochs": 100,
            # "batch_size": 64
        })
        return cfg


    def plan(self, ego_veh_state:elm.VehicleState, obstacles:elm.TrackingBoxList, routed_local_map:elm.RoutedLocalMap)\
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:

        if not (routed_local_map is None):
            self.set_local_map(routed_local_map)

        state = self.state_encoder(ego_veh_state, obstacles, self.local_map).unsqueeze(0)
        action = self.act(state.to(self.device)).squeeze(0)

        traj = self.action_decoder(action.detach().cpu(), ego_veh_state, obstacles, self.local_map)
        return traj


if __name__ == '__main__':
    from spider.interface import DummyBenchmark
    #
    # planner = MlpPlanner()
    #
    # # obs = DummyBenchmark.get_environment_presets()
    # # traj = planner.plan(*obs)
    #
    # bm = DummyBenchmark()
    # bm.test(planner)


    pass
