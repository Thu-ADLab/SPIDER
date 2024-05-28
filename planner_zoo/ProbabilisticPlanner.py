import math
from typing import Union

import numpy as np
import torch.nn as nn
import torch


import spider.elements as elm
from spider.sampler.LatticeSampler import LatticeSampler

from spider.rl.state.StateConverter import KineStateEncoder
from spider.rl.action.ActionConverter import DiscreteTrajActionDecoder, DiscreteTrajActionEncoder
from spider.rl.policy.ClassificationILPolicy import ClassificationILPolicy
import spider.rl.convert as cvt
from spider.planner_zoo.BaseNeuralPlanner import BaseNeuralPlanner
from spider.utils.transform.frenet import FrenetTransformer

class MlpCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super().__init__()
        self.mlp = nn.Sequential(  # 5-layer MLP
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_dim)#, nn.Sigmoid()
        )

    def forward(self, x):
        return torch.nn.functional.softmax(self.mlp(x), dim=-1)


class ProbabilisticPlanner(BaseNeuralPlanner):
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
            calc_by_need=True
        )

        self.action_decoder = DiscreteTrajActionDecoder(sampler=self.trajectory_sampler)
        self.action_encoder = DiscreteTrajActionEncoder(sampler=self.trajectory_sampler)
        # self.action_encoder = cvt.Compose(
        #     cvt.TrajToTensor(),
        #     cvt.ZeroOffset(dim=-2),
        # )

        self.policy = ClassificationILPolicy(
            MlpCritic(self.state_encoder.state_dim, self.action_decoder.action_dim).to(self.device),
            lr = self.config["learning_rate"],
            enable_tensorboard=self.config["enable_tensorboard"],
            tensorboard_root=self.config["tensorboard_root"]
        )


    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "steps": 20,
            "dt": 0.2,

            "num_object": 5,
            "normalize": False,
            "relative": False,
            "longitudinal_range": (-50, 100),
            "lateral_range": (-20,20),

            "end_s_candidates": (10, 20, 40, 60),
            "end_l_candidates": (-3.5, 0, 3.5),
            "end_v_candidates": tuple(i * 60 / 3.6 / 3 for i in range(4)),
            "end_T_candidates": (2, 4, 8),

            "learning_rate": 0.0001,
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

        state = self.state_encoder(ego_veh_state, obstacles, self.local_map)
        action = self.act(state.to(self.device))

        traj = self.action_decoder(action.detach().cpu(), ego_veh_state, obstacles, self.local_map)
        return traj

    ##############################################
    # def build_frenet_lane(self, target_lane_idx):
    #     if target_lane_idx < 0 or target_lane_idx >= len(self.local_map.lanes):
    #         raise ValueError("Invalid target lane record_index")
    #     target_lane = self.local_map.lanes[target_lane_idx]
    #     self.coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)
    #
    # def sample_candidate_trajectories(self, ego_veh_state:elm.VehicleState):
    #     # 把自车位置匹配到对应车道，并且把自车点位转换为Frenet坐标
    #     # 坐标系建立及坐标转换（车道匹配+车道决策+坐标转换）
    #     ego_lane_idx = self.local_map.match_lane(ego_veh_state.x(), ego_veh_state.y())  # self.match_lanes(ego_veh_state)  # 把自车位置匹配到对应车道
    #     target_lane_idx = ego_lane_idx  # 目前车道决策还没写上，先默认自车车道，按道理是一个以自车车道输入的函数
    #     self.build_frenet_lane(target_lane_idx)
    #     fstate_start = self.coordinate_transformer.cart2frenet(ego_veh_state.x(), ego_veh_state.y(),
    #                                                            ego_veh_state.v(), ego_veh_state.yaw(),
    #                                                            ego_veh_state.a(), ego_veh_state.kappa(), order=2)
    #     candidate_trajectories = self.trajectory_sampler.sample(
    #         (fstate_start.s, fstate_start.s_dot, fstate_start.s_2dot),
    #         (fstate_start.l, fstate_start.l_prime, fstate_start.l_2prime)
    #     )
    #     return candidate_trajectories


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
