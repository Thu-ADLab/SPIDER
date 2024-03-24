import math
from typing import Union

import torch.nn as nn
import torch


import spider.elements as elm
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
        return torch.tanh(self.fc3(x))


class MlpPlanner(BaseNeuralPlanner):
    def __init__(self, config, actor:nn.Module=None):
        super(MlpPlanner, self).__init__(config)

        self.obs_feat_num = 7
        self.state_dim = self.config["obs_veh_num"] * self.obs_feat_num
        self.action_dim = 2 * self.steps

        self.actor = None
        if self.actor is None:
            self.actor = MlpActor(self.state_dim, self.action_dim).to(self.device)
        else:
            self.actor: nn.Module = actor.to(self.device)

        self._roi_radius_square = self.config["roi_radius"] ** 2


    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "obs_veh_num": 5,
            "roi_radius": 100,
            "padding_value": 0.0,
        })
        return cfg


    # 后面关于状态和动作变换的内容都塞到状态和动作本身的文件里面去。
    def state_transform(self, ego_veh_state:elm.VehicleState, obstacles:elm.TrackingBoxList, routed_local_map:elm.RoutedLocalMap) \
            -> torch.Tensor:
        '''
        x, y, length, width, yaw, vx, vy
        '''
        # 自车和他车数据，agent
        ego_x, ego_y = ego_veh_state.x(), ego_veh_state.y()
        ego_yaw = ego_veh_state.yaw()
        ego_vx = ego_veh_state.v() * math.cos(ego_yaw)
        ego_vy = ego_veh_state.v() * math.sin(ego_yaw)
        ego_state = [*ego_veh_state.obb, ego_vx, ego_vy]

        obstacle_state = []
        obstacles = obstacles.sort_by_dist(ego_x, ego_y)
        for tb in obstacles:
            tb: elm.TrackingBox
            if tb.x**2 + tb.y ** 2 < self._roi_radius_square:
                obstacle_state.append([*tb.obb, tb.vx, tb.vy])

        agents_state = ego_state + obstacle_state
        agents_state = self._fixed_veh_num(agents_state)

        agents_state = torch.tensor(agents_state, dtype=torch.float)


        # todo: concat当前地图信息
        state = agents_state

        return state.view(-1, self.state_dim).to(self.device)

    def action_transform(self, action: torch.Tensor) -> Union[elm.Trajectory, elm.FrenetTrajectory]:
        pass


    def _fixed_veh_num(self, agents_state: Union[torch.Tensor, list]):
        '''
        agents_state should not be batched.  dim [veh_num, feat_num]
        '''
        # origin_size = agents_state.shape
        # agents_state = agents_state.view(-1, self.obs_feat_num, self.obs_feat_num)

        delta_veh_num = self.config["obs_veh_num"] - len(agents_state)
        if delta_veh_num > 0: # self.config["obs_veh_num"] > len(agents_state)
            # padding
            if isinstance(agents_state, torch.Tensor):
                padding_state = torch.full((delta_veh_num, self.obs_feat_num),
                           self.config["padding_value"], dtype=torch.float).to(agents_state.device)
                agents_state = torch.cat([agents_state, padding_state])
            elif isinstance(agents_state, list):
                padding_state = [self.config["padding_value"]] * self.obs_feat_num
                padding_state = [padding_state] * delta_veh_num
                agents_state.extend(padding_state)
        elif delta_veh_num < 0:
            # truncate
            agents_state = agents_state[:self.config["obs_veh_num"]]

        return agents_state




#

