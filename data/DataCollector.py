import torch
import os
import math
from typing import Union

import spider
import spider.elements as elm

import tqdm



class DataCollector:
    # 要重新写，跟interface配合很不好。应该就是利用interface随机初始化环境。然后一直执行。
    def __init__(self,
                 planner: spider.planner_zoo.BasePlanner,
                 interface: spider.interface.BaseInterface=None,
                 data_root='./dataset/',
                 ):

        self.planner = planner
        self.interface = interface
        self.data_root = data_root

        if not os.path.exists(self.data_root):
            os.mkdir(self.data_root)
        self.data = []
        # 以后可以有多个planner参与收集数据!
        # self.experience = []

    def set_planner(self, planner):
        self.planner = planner

    def state_transform(self, ego_veh_state: elm.VehicleState, obstacles: elm.TrackingBoxList,
                        routed_local_map: elm.RoutedLocalMap) \
            -> torch.Tensor:
        '''
        x, y, length, width, yaw, vx, vy
        '''
        self._roi_radius_square = 10000
        self.device = torch.device("cpu")
        self.state_dim = 7*5


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
            if tb.x ** 2 + tb.y ** 2 < self._roi_radius_square:
                obstacle_state.append([*tb.obb, tb.vx, tb.vy])

        agents_state = [ego_state] + obstacle_state
        agents_state = self._fixed_veh_num(agents_state)

        agents_state = torch.tensor(agents_state, dtype=torch.float)

        # todo: concat当前地图信息
        state = agents_state

        return state.view(-1, self.state_dim).to(self.device)

    def _fixed_veh_num(self, agents_state: Union[torch.Tensor, list]):

        self.config = {"obs_veh_num": 5, "padding_value": 0.0}
        self.obs_feat_num = 7

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

    # def action_transform(self, action: torch.Tensor, ego_state:elm.VehicleState) -> Union[elm.Trajectory, elm.FrenetTrajectory]:
    #     lon_range = [-30,100],
    #     lat_range = [-30,30]
    #
    #     action = action.view(2, -1)
    #     xs, ys = action
    #     xs


    def collect(self, num_frames=10000):
        # count = 0
        self.interface.reset() # 这里interface的reset函数输出还没统一，目前只支持dummyinterface

        # while count < num_frames:
        for i in tqdm.tqdm(range(num_frames)):
            ego, obs, lmap = self.interface.wrap_observation()
            done = self.interface.is_done()
            self.data.append([self.state_transform(ego, obs, lmap), done])
            # count += 1

            if done:
                self.interface.reset()
                continue

            traj = self.planner.plan(ego, obs, lmap)
            if traj is None:
                self.interface.reset()
                continue

            self.interface.conduct_trajectory(traj)
            # count += 1

        torch.save(self.data, self.data_root+'dataset.pt')




