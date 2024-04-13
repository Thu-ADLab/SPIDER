import torch
import torch.nn as nn
from typing import Union
from abc import abstractmethod

import spider
import spider.elements as elm
from spider.planner_zoo.BasePlanner import BasePlanner


class BaseNeuralPlanner(BasePlanner):
    def __init__(self, config=None, policy:nn.Module=None, state_encoder:nn.Module=None, action_decoder:nn.Module=None):
        super(BaseNeuralPlanner, self).__init__(config)

        # self.state_dim = 0
        # self.action_dim = 0

        self.state_encoder: nn.Module = state_encoder  # Observation -> tensor (state)
        self.action_decoder: nn.Module = action_decoder  # tensor(state) -> tensor (action)
        self.policy: nn.Module = policy  # tensor (action) -> Plan

        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if self.policy is not None:
            self.policy.to(self.device)


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = super().default_config()
        config.update({
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 15,
            "dt": 0.2,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "print_info": True,

            "model_path": './model.pth' # todo:以后加上若干epoch自动保存的功能
        })
        return config

    @property
    def training(self):
        return self.policy.training


    def act(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.training:
            action = self.policy(state)
        else:
            with torch.no_grad():
                action = self.policy(state)
        return action

    def plan(self, ego_veh_state:elm.VehicleState, obstacles:elm.TrackingBoxList, routed_local_map:elm.RoutedLocalMap)\
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        state = self.state_encoder(ego_veh_state, obstacles, routed_local_map).unsqueeze(0)
        action = self.act(state.to(self.device)).squeeze(0)
        traj = self.action_decoder(action.detach().cpu(), ego_veh_state, obstacles, routed_local_map)
        return traj

    def to(self, device):
        self.device = device
        self.policy.to(device)

    def configure(self, config: dict):
        raise RuntimeError("Neural planner does not support. Re-instantiate a planner instead! ")


    def load_state_dict(self, path):
        path = self.config["model_path"] if path is None else path
        self.policy.load_state_dict(torch.load(path))

    def save_state_dict(self, path=None):
        path = self.config["model_path"] if path is None else path
        torch.save(self.policy.state_dict(), path)

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()

    def set_action_decoder(self, action_decoder:nn.Module):
        self.action_decoder = action_decoder

    def set_state_encoder(self, state_encoder:nn.Module):
        self.state_encoder = state_encoder

    def set_policy(self, policy: nn.Module):
        self.policy = policy
        self.policy.to(self.device)


