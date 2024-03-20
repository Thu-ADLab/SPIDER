import torch
import torch.nn as nn
from typing import Union
from abc import abstractmethod

import spider
import spider.elements as elm
from spider.planner_zoo.BasePlanner import BasePlanner


class BaseActorPlanner(BasePlanner):
    '''
    actor网络储存一个直接从状态到动作的映射
    '''
    def __init__(self, config, actor: nn.Module=None):
        super(BaseActorPlanner, self).__init__(config)

        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = None

        if actor is not None:
            self.actor: nn.Module = actor.to(self.device)


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = super().default_config()
        config.update({
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 50,
            "dt": 0.1,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "print_info": True,
        })
        return config

    @abstractmethod
    def state_transform(self, ego_veh_state, obstacles, local_map, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def action_transform(self, action: torch.Tensor, *args, **kwargs)\
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        pass

    @abstractmethod
    def act(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        action = self.actor(state.to(self.device))
        return action

    def plan(self, ego_veh_state:elm.VehicleState, obstacles:elm.TrackingBoxList, routed_local_map:elm.RoutedLocalMap)\
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        state = self.state_transform(ego_veh_state, obstacles, routed_local_map)
        action = self.act(state)
        traj = self.action_transform(action)
        return traj

    def to(self, device):
        self.actor.to(device)

    def configure(self, config: dict):
        raise RuntimeError("ML planner does not support. Re-instantiate a planner instead! ")


class BaseCriticPlanner(BasePlanner):
    '''
    critic储存一个给轨迹打分的网络
    critic planner类似于基于采样的planner，采样一堆轨迹，然后选择得分最高的
    '''

    def __init__(self, config, critic: nn.Module = None):
        super(BaseCriticPlanner, self).__init__(config)

        self.sampler = None
        self.action_candidates = None

        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.critic = None

        if critic is not None:
            self.critic: nn.Module = critic.to(self.device)

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = super().default_config()
        config.update({
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 50,
            "dt": 0.1,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "print_info": True,
        })
        return config

    @abstractmethod
    def state_transform(self, ego_veh_state, obstacles, local_map, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def action_transform(self, action: torch.Tensor, *args, **kwargs) \
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        pass

    @abstractmethod
    def criticize(self, action, state: torch.Tensor=None, *args, **kwargs):
        pass

    def plan(self, ego_veh_state: elm.VehicleState, obstacles: elm.TrackingBoxList,
             routed_local_map: elm.RoutedLocalMap) \
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        pass

    def to(self, device):
        self.critic.to(device)

    def configure(self, config: dict):
        raise RuntimeError("ML planner does not support. Re-instantiate a planner instead! ")


class BaseActorCriticPlanner(BasePlanner):
    '''
    actor网络储存一个直接从状态到(一个或一堆)动作的映射,
    critic储存一个给轨迹打分的网络
    critic planner类似于基于采样的planner，采样一堆轨迹，然后选择得分最高的
    '''

    def __init__(self, config, actor: nn.Module = None, critic: nn.Module = None):
        super(BaseActorCriticPlanner, self).__init__(config)

        self.device = self.config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.actor = None
        if actor is not None:
            self.actor: nn.Module = actor.to(self.device)

        self.action_candidates = None

        self.critic = None
        if critic is not None:
            self.critic: nn.Module = critic.to(self.device)


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = super().default_config()
        config.update({
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 50,
            "dt": 0.1,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "print_info": True,
        })
        return config

    @abstractmethod
    def state_transform(self, ego_veh_state, obstacles, local_map, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def action_transform(self, action: torch.Tensor, *args, **kwargs) \
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        pass

    @abstractmethod
    def act(self, state: torch.Tensor=None, *args, **kwargs):
        pass

    @abstractmethod
    def criticize(self, action, state: torch.Tensor=None, *args, **kwargs):
        pass

    def plan(self, ego_veh_state: elm.VehicleState, obstacles: elm.TrackingBoxList,
             routed_local_map: elm.RoutedLocalMap) \
            -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        pass

    def to(self, device):
        self.critic.to(device)

    def configure(self, config: dict):
        raise RuntimeError("ML planner does not support. Re-instantiate a planner instead! ")

