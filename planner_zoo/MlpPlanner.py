import math
from typing import Union

import torch.nn as nn
import torch


import spider.elements as elm
from spider.rl.state.StateConverter import KineStateEncoder
from spider.rl.action.ActionConverter import TrajActionDecoder, TrajActionEncoder
from spider.rl.policy.RegressionILPolicy import RegressionILPolicy
from spider.planner_zoo.BaseNeuralPlanner import BaseNeuralPlanner


class MlpActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(MlpActor, self).__init__()
        self.mlp = nn.Sequential(  # 5-layer MLP
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class MlpPlanner(BaseNeuralPlanner):
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

        self.policy = RegressionILPolicy(
            MlpActor(self.state_encoder.state_dim, self.action_decoder.action_dim).to(self.device),
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
            "normalize": False,
            "relative": False,
            "longitudinal_range": (-50, 100),
            "lateral_range": (-20,20),

            "learning_rate": 0.0001,
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


class MlpPlanner_(BaseNeuralPlanner):
    '''
    旧版，另一种实现训练的手段
    现在弃用
    '''
    def __init__(self, config=None):
        super().__init__(config)

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

    def learn(self, dataloader):
        import tqdm
        import torch.optim as optim
        import matplotlib.pyplot as plt

        criterion = torch.nn.MSELoss()  # torch.nn.L1Loss()
        optimizer = optim.Adam(planner.policy.parameters(), lr=0.0001)

        planner.train_mode()
        avg_losses = []
        for epoch in tqdm.tqdm(range(100)):
            temp_losses = []
            for i, exp in enumerate(dataloader):
                states, actions = exp[:2]
                states = states.to(planner.device)
                actions = actions.to(planner.device)

                pred_actions = planner.policy(states)
                loss = criterion(actions, pred_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                temp_losses.append(loss.item())

            avg_losses.append(sum(temp_losses) / len(temp_losses))
            plt.cla()
            plt.plot(avg_losses)
            plt.pause(0.001)
        plt.savefig("mlp_train.png")
        plt.close()


if __name__ == '__main__':
    from spider.interface import DummyBenchmark

    planner = MlpPlanner()

    # obs = DummyBenchmark.get_environment_presets()
    # traj = planner.plan(*obs)

    bm = DummyBenchmark()
    bm.test(planner)


    pass
