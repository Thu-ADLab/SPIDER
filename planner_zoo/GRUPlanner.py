import math
from typing import Union

import torch.nn as nn
import torch

import spider.elements as elm
from spider.rl.state.StateConverter import KineStateEncoder
from spider.rl.action.ActionConverter import TrajActionDecoder, TrajActionEncoder
from spider.rl.policy.RegressionImitationPolicy import RegressionImitationPolicy
from spider.planner_zoo.BaseNeuralPlanner import BaseNeuralPlanner

class GRUActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, lon_range=(-80, 80), lat_range=(-80, 80)):
        super().__init__()
        assert output_dim % 2 == 0
        self.traj_steps = int(output_dim / 2)

        self.backbone = nn.Sequential(  # MLP，提取特征
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )  # .to(self.device)
        self.rnn_cell = nn.GRUCell(input_size=2, hidden_size=hidden_size)
        # self.decode_wp = nn.Linear(hidden_size, 2)
        self.decode_wp = nn.Sequential(  # MLP，提取特征
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

        # waypoint 0 for the action decoder
        # self._lon_range = lon_range
        self._waypoint0 = torch.FloatTensor([
            (0.0 - lon_range[0]) / (lon_range[1] - lon_range[0]), # 0点位置
            (0.0 - lat_range[0]) / (lat_range[1] - lat_range[0])
        ])

    def forward(self, state):
        batch_size = state.shape[:-1]

        z = self.backbone(state)
        output_wp = []
        traj_hidden_state = []
        # 初始输入
        x = torch.zeros(size=(*batch_size, 2), dtype=z.dtype).type_as(z)
        x[..., :] = self._waypoint0
        # 轨迹的自回归生成
        for _ in range(self.traj_steps):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x
            z = self.rnn_cell(x_in, z)
            traj_hidden_state.append(z)
            dx = self.decode_wp(z)
            x = dx + x
            output_wp.append(x)  # 可以改成torch.cat

        stack_dim = len(batch_size)
        outputs_trajectory = torch.stack(output_wp, dim=stack_dim).flatten(-2, -1)
        outputs_trajectory = torch.clamp(outputs_trajectory,0.,1.)
        return outputs_trajectory


class GRUPlanner(BaseNeuralPlanner):
    def __init__(self, config=None):
        super().__init__(config)

        self.state_encoder = KineStateEncoder(
            normalize=self.config["normalize"],
            relative=self.config["relative"],
            num_object=self.config["num_object"],
            x_range=self.config["longitudinal_range"],
            y_range=self.config["lateral_range"])

        self.action_decoder = TrajActionDecoder(self.config["steps"], self.config["dt"],
                                                lon_range=self.config["longitudinal_range"], lat_range=self.config["lateral_range"])

        self.action_encoder = TrajActionEncoder(lon_range=self.config["longitudinal_range"], lat_range=self.config["lateral_range"])

        actor = GRUActor(self.state_encoder.state_dim, self.action_decoder.action_dim,
                         lon_range=self.config["longitudinal_range"], lat_range=self.config["lateral_range"])
        self.policy = RegressionImitationPolicy(
            actor.to(self.device),
            criterion = nn.MSELoss(),
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
            "num_object": 6,
            "longitudinal_range": (-30, 60),
            "lateral_range": (-10,10),
            "normalize": False,
            "relative": False,

            "learning_rate": 0.00001,
            "enable_tensorboard": False,
            "tensorboard_root": './tensorboard/'
            # "epochs": 100,
            # "batch_size": 64
        })
        return cfg


    def plan(self, ego_veh_state:elm.VehicleState, obstacles:elm.TrackingBoxList, routed_local_map:elm.RoutedLocalMap)\
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        state = self.state_encoder(ego_veh_state, obstacles, routed_local_map)
        action = self.act(state.to(self.device))
        traj = self.action_decoder(action.detach().cpu(), ego_veh_state)
        return traj