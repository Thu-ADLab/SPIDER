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
    def __init__(self, input_dim, output_dim, mlp_hidden_dim=64, num_gru_layers=1, gru_hidden_dim=32,
                 lon_range=(-80, 80), lat_range=(-80, 80)):
        super().__init__()
        assert output_dim % 2 == 0
        self.traj_steps = int(output_dim / 2)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers

        self.backbone = nn.Sequential(  # MLP，提取特征
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, gru_hidden_dim * self.num_gru_layers),
        )  # .to(self.device)

        self.gru = nn.GRU(2, gru_hidden_dim, num_layers=self.num_gru_layers)
        # self.rnn_cell = nn.GRUCell(input_size=2, mlp_hidden_dim=mlp_hidden_dim,) # 这个不支持多层的

        # self.decode_wp_delta = nn.Linear(gru_hidden_dim, 2)
        self.decode_wp_delta = nn.Sequential(  # MLP，提取特征
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(gru_hidden_dim, 2),
        )

        # waypoint 0 for the action decoder
        # self._lon_range = lon_range
        self._waypoint0 = torch.FloatTensor([
            (0.0 - lon_range[0]) / (lon_range[1] - lon_range[0]), # 0点位置
            (0.0 - lat_range[0]) / (lat_range[1] - lat_range[0])
        ])

    # def forward(self, state):
    #     if len(state.shape) == 1:
    #         state = state.unsqueeze(0)
    #
    #     batch_size, feat_dim = state.shape
    #
    #     z = self.backbone(state)
    #     ####
    #     z = z.view(self.num_gru_layers, batch_size, self.gru_hidden_dim)
    #     ####
    #     # 初始输入
    #     out = torch.zeros(size=(1, batch_size, 2), dtype=z.dtype).type_as(z)
    #     out[..., :] = self._waypoint0 # sequence_length, batch size, input_size
    #     # traj_hidden_state = []
    #     ##
    #     ##
    #     # 轨迹的自回归生成
    #     for _ in range(self.traj_steps-1):
    #         # x_in = torch.cat([x, target_point], dim=1)
    #         # z = self.gru(x_in, z)
    #
    #         next_seq, z = self.gru(out, z)
    #         # traj_hidden_state.append(z)
    #
    #         next_wp = self.decode_wp_delta(next_seq[-1:]) + out[-1:, ...]
    #
    #         out = torch.cat([out, next_wp], dim=0)
    #
    #         # dx = self.decode_wp_delta(out)
    #         # x = dx + x
    #         # output_wp.append(x)  # 可以改成torch.cat
    #         # output_wp = torch.cat([output_wp, x], dim=0)
    #
    #     # stack_dim = len(batch_size)
    #     # outputs_trajectory = torch.stack(output_wp, dim=stack_dim).flatten(-2, -1)
    #     out = out.transpose(0,1).contiguous().view(batch_size, -1)
    #     # outputs_trajectory = output_wp.flatten(-2, -1)
    #     out = torch.clamp(out,0.,1.)
    #     return out


    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            squeeze_flag = True
        else:
            squeeze_flag = False
        batch_size, feat_dim = state.shape

        z = self.backbone(state)
        ####
        z = z.view(self.num_gru_layers, batch_size, self.gru_hidden_dim)
        ####
        # 初始输入
        x = torch.zeros(size=(batch_size, 2), dtype=z.dtype).type_as(z)
        x[..., :] = self._waypoint0
        output_wp = [x]
        # traj_hidden_state = []
        ##
        x = x.unsqueeze(0)
        ##
        # 轨迹的自回归生成
        for _ in range(self.traj_steps-1):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x
            # z = self.gru(x_in, z)
            out, z = self.gru(x_in, z)

            dx = self.decode_wp_delta(out)
            x = dx + x

            output_wp.append(x.squeeze(0))

        outputs_trajectory = torch.stack(output_wp, dim=1).flatten(-2, -1)
        outputs_trajectory = torch.clamp(outputs_trajectory,0.,1.)

        if squeeze_flag:
            outputs_trajectory = outputs_trajectory.squeeze(0)

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
            # criterion = nn.MSELoss(),
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