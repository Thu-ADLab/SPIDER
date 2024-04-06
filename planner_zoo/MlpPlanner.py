import math
from typing import Union

import torch.nn as nn
import torch


import spider.elements as elm
from spider.rl.state.StateConverter import KineStateEncoder
from spider.rl.action.ActionConverter import TrajActionDecoder, TrajActionEncoder
from spider.planner_zoo.BaseNeuralPlanner import BaseNeuralPlanner


class MlpActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(MlpActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class MlpPlanner(BaseNeuralPlanner):
    def __init__(self, config=None):
        super(MlpPlanner, self).__init__(config)

        self.state_encoder = KineStateEncoder(num_object=self.config["num_object"],
                                              x_range=self.config["longitudinal_range"],
                                              y_range=self.config["lateral_range"])

        self.action_decoder = TrajActionDecoder(self.config["steps"], self.config["dt"],
                                                lon_range=self.config["longitudinal_range"], lat_range=self.config["lateral_range"])

        self.action_encoder = TrajActionEncoder(lon_range=self.config["longitudinal_range"], lat_range=self.config["lateral_range"])

        self.policy = MlpActor(self.state_encoder.state_dim, self.action_decoder.action_dim).to(self.device)


    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "steps": 20,
            "dt": 0.2,
            "num_object": 6,
            "longitudinal_range": (-30, 60),
            "lateral_range": (-30,30)
        })
        return cfg


    def plan(self, ego_veh_state:elm.VehicleState, obstacles:elm.TrackingBoxList, routed_local_map:elm.RoutedLocalMap)\
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        state = self.state_encoder(ego_veh_state, obstacles, routed_local_map)
        action = self.act(state.to(self.device))
        traj = self.action_decoder(action.detach().cpu(), ego_veh_state)
        return traj


if __name__ == '__main__':
    from spider.interface import DummyBenchmark

    planner = MlpPlanner()

    # obs = DummyBenchmark.get_environment_presets()
    # traj = planner.plan(*obs)

    bm = DummyBenchmark()
    bm.test(planner)


    pass
