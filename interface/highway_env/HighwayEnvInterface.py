import math
from typing import Union, Tuple, Type
import highway_env
import gymnasium as gym
import highway_env.envs
import numpy as np

import spider
import spider.elements as elm
from spider.elements import TrackingBoxList, OccupancyGrid2D, RoutedLocalMap, VehicleState, Trajectory

from spider.control import SimpleController

# from spider.elements import Location, Rotation, Transform,

'''
Interface逻辑：
observation -> perception, routed_local_map, localization (planner统一的输入表达) ->(planner)-> output -> action
'''


class HighwayEnvInterface:
    def __init__(self,
                 env: Union[highway_env.envs.AbstractEnv, gym.Env], # 这个union纯粹是为了避免IDE报提示
                 # observation_flag=spider.HIGHWAYENV_OBS_KINEMATICS,
                 perception_flag=spider.PERCEPTION_BOX,
                 output_flag=spider.OUTPUT_TRAJECTORY,
                 veh_length=5.0,
                 veh_width=2.0):

        self._env = env # 注意，由于python是引用传递，所以这个_env完全等价于外部的env
        self._env_config = env.unwrapped.config

        # 车长车宽没考虑

        # highway-env规范的输入输出flag
        self.observation_flag = self._observation_type()
        self.action_flag = self._action_type()

        # spider planner规范的输入输出flag
        self.perception_flag = perception_flag
        self.output_flag = output_flag

        self.veh_length = veh_length
        self.veh_width = veh_width

        # self.kine_feature_index_mapping = self._build_feature_index_mapping()
        if self.observation_flag == spider.HIGHWAYENV_OBS_KINEMATICS:
            self._all_features = self._env.unwrapped.observation_type.FEATURES
            assert "x" in self._all_features and "y" in self._all_features

        self._routed_local_map = None

        self._controller = SimpleController()

    # def observe_env(self):
    #     obs, info = self._env.
    @property
    def observation(self):
        return self._env.observation_type.observe()

    # @staticmethod
    # def calc_reward(highway_env, observation, action):
    #     return reward

    def reset(self):
        # todo: qzl: 想一想有没有什么更多的需要reset的
        self._routed_local_map = None
        return self._env.reset()

    def step(self, action) -> Tuple["Observation", float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        return self._env.step(action)

    def wrap_observation(self, observation) \
            -> Tuple[VehicleState, Union[TrackingBoxList, OccupancyGrid2D], RoutedLocalMap]:
        '''
        qzl:这个函数的名字可以再改
        把observation改成planner的统一输入形式。
        '''

        if self.observation_flag == spider.HIGHWAYENV_OBS_KINEMATICS:
            # highway-env的观测是kinematics

            ego_veh_state = self._get_kine_ego_state(observation)

            if self.perception_flag == spider.PERCEPTION_BOX:
                perception = self._wrap_kine2box(observation)
            elif self.perception_flag == spider.PERCEPTION_OCC:
                raise NotImplementedError("not supported now...")
                # perception = self._wrap_kine2box(observation)
            else:
                raise ValueError("INVALID perception_flag")

        else:
            raise NotImplementedError("not supported now...")

        if self._need_to_update_map():
            local_map = self.extract_map()
        else:
            local_map = self._routed_local_map

        return ego_veh_state, perception, local_map

    def convert_to_action(self, planner_output, ego_veh_state:VehicleState=None):
        if planner_output is None:
            raise AssertionError("The planner outputs NO results. Please check whether it can find a valid solution, and it is recommended to add a fallback trajectory generation scheme.")

        if self.action_flag != spider.HIGHWAYENV_ACT_CONTINUOUS:
            raise AssertionError("not supported now. Please change the environment action type into ContinuousAction.")

        if self.output_flag == spider.OUTPUT_TRAJECTORY:  # 轨迹
            planner_output:Trajectory
            acc, steer = self._controller.get_control(planner_output, ego_veh_state)
            return acc, steer
        else:  # 控制量
            raise NotImplementedError("control not supported now...")

    def conduct_trajectory(self, trajectory:Trajectory, ego_veh_state:VehicleState=None):
        action = self.convert_to_action(trajectory, ego_veh_state)
        return self._env.step(action)

    def conduct_output(self, planner_output, ego_veh_state:VehicleState=None):
        '''
        qzl: 应该写成直接执行动作呢还是写成输出对应格式的动作呢？
        '''
        if self.output_flag == spider.OUTPUT_TRAJECTORY:  # 轨迹
            return self.conduct_trajectory(planner_output, ego_veh_state)
        else:  # 控制量
            raise NotImplementedError("not supported now...")

    def _observation_type(self):
        type_str = self._env_config["observation"]["type"]
        if type_str == "Kinematics":
            return spider.HIGHWAYENV_OBS_KINEMATICS
        elif type_str == "GrayscaleObservation":
            return spider.HIGHWAYENV_OBS_GRAYIMG
        elif type_str == "OccupancyGird":
            return spider.HIGHWAYENV_OBS_OCCUPANCY
        elif type_str == "TimeToCollision":
            return spider.HIGHWAYENV_OBS_TTC
        else:
            raise ValueError("INVALID observation type")

    def _action_type(self):
        type_str = self._env_config["action"]["type"]
        if type_str == "ContinuousAction":
            return spider.HIGHWAYENV_ACT_CONTINUOUS
        elif type_str == "DiscreteAction":
            return spider.HIGHWAYENV_ACT_DISCRETE
        elif type_str == "DiscreteMetaAction":
            return spider.HIGHWAYENV_ACT_META
        else:
            raise ValueError("INVALID action type")

    # def _build_feature_index_mapping(self):
    #     features: list = self._env_config["observation"]["features"]
    #     necessary_keys = ["presence", "x", "y", "vx", "vy","heading","cos_h", "sin_h"]
    #     for key in necessary_keys:
    #         if key in features:
    #             idx = features.index(key)
    #             self.kine_feature_index_mapping[key] = idx
    #     assert "x" in self.kine_feature_index_mapping and "y" in self.kine_feature_index_mapping

    def _get_veh_info_dict(self, veh_info_vector, feat_names, *, calc_heading: bool = True) -> dict:
        '''
        feat_names: 需要给出的feature的名字
        veh_info_vector: 目标车辆的Observation的对应切片
        '''
        # todo:qzl: 要加入，关于absolute和normalize的自调整

        # idxs = self.kine_feature_index_mapping
        #
        # features: list = self._env_config["observation"]["features"]

        all_veh_info_dict = dict(zip(self._all_features, veh_info_vector))

        # veh_info_dict = {key: veh_info_vector[idxs[key]] if key in features else 0.0 for key in feat_names}
        veh_info_dict = {key: all_veh_info_dict[key] if key in all_veh_info_dict else 0.0
                         for key in feat_names}

        if calc_heading:
            if "heading" in feat_names:  # 如果需要储存heading信息，但heading可能用别的方式表达，那么需要进行以下处理
                if "heading" in self._all_features:
                    pass  # 如果本身observation给出的heading就直接用heading信息
                elif "cos_h" in self._all_features:
                    veh_info_dict["heading"] = math.acos(all_veh_info_dict["cos_h"])
                elif "sin_h" in self._all_features:
                    veh_info_dict["heading"] = math.asin(all_veh_info_dict["sin_h"])
                elif ("vx" in self._all_features) and ("vy" in self._all_features):
                    veh_info_dict["heading"] = math.atan2(all_veh_info_dict["vy"], all_veh_info_dict["vx"])
                else:
                    pass  # 默认设0.0

        return veh_info_dict

    def _get_kine_ego_state(self, observation) -> VehicleState:
        necessary_feat = ["x", "y", "vx", "vy", "heading", "cos_h", "sin_h"]
        ego_info: dict = self._get_veh_info_dict(observation[0], necessary_feat)
        loc = elm.Location(ego_info["x"], ego_info["y"], 0.)
        rot = elm.Rotation(0., ego_info["heading"], 0.)
        velocity = elm.Vector3D(ego_info["vx"], ego_info["vy"], 0.)
        ego_state = VehicleState(elm.Transform(loc, rot), velocity, elm.Vector3D())
        return ego_state

    def _wrap_kine2box(self, observation) -> TrackingBoxList:

        necessary_feat = ["presence", "x", "y", "vx", "vy", "heading"]
        tbox_list = TrackingBoxList()
        for i, veh_info_vector in enumerate(observation):
            if i == 0:
                # i=0 是自车，不需要加入BoundingBox
                continue

            veh_info = self._get_veh_info_dict(veh_info_vector, necessary_feat)
            if "presence" in self._all_features and (not veh_info["presence"]):
                # 存在presence属性，且该车的presence属性是0，表明这不存在
                continue

            x, y = veh_info["x"], veh_info["y"]
            heading = veh_info["heading"]
            vx = veh_info["vx"]
            vy = veh_info["vy"]
            tbox = elm.TrackingBox(obb=(x, y, self.veh_length, self.veh_width, heading), vx=vx, vy=vy)

            tbox_list.append(tbox)

        return tbox_list

    def _wrap_kine2grid(self) -> OccupancyGrid2D:
        pass

    ############## 地图信息 #############
    def _need_to_update_map(self) -> bool:
        if self._routed_local_map is None:
            return True
        return True # todo: qzl: 现在还没来得及写，默认是每一刻都需要更新，后面要加上这个判断的逻辑

    def extract_map(self, start_s=0.0, delta_s=1.0) -> RoutedLocalMap:
        # todo:qzl:现在没有加入extend车道的功能，也就是在长度不够的情况下，通过route信息，把下一个路段的lane加载进来。
        # todo:qzl: 另外导航信息也没加到routedmap里面
        local_map = RoutedLocalMap()

        ego_veh = self._env.unwrapped.vehicle

        network = ego_veh.road.network

        ego_lane_idx = ego_veh.lane_index
        all_neighbor_lanes = network.all_side_lanes(ego_lane_idx)

        for i, lane_idx in enumerate(all_neighbor_lanes):
            lane = network.get_lane(lane_idx)
            # qzl: 在highway-env里面，如果不加min，lane.length是10000，后面的三次样条插值什么的计算极其复杂
            # 先制作自己感兴趣的区间
            roi_start = max([0., start_s-50])
            roi_length = min([200, lane.length-roi_start])
            roi_end = roi_length + roi_start

            sampled_s = np.arange(roi_start, roi_end, delta_s)
            center_line = [lane.position(s, 0) for s in sampled_s]
            spd_lane = elm.Lane(i, center_line, width=lane.width_at(0), speed_limit=lane.speed_limit)
            local_map.add_lane(spd_lane)

        # local_map.network = ego_veh.road.network # todo:qzl: 由于没有确认好network的形式，建议后面再说
        # local_map.route = ego_veh.route if not (ego_veh.route is None) else [target_lane_idx]

        # qzl: 以后要加上延长的逻辑
        return local_map


if __name__ == '__main__':
    # import highway_env
    from matplotlib import pyplot as plt

    # env = gym.make('highway-v0', render_mode='rgb_array')
    env = gym.make("highway-v0")
    env.unwrapped.configure({
        "show_trajectories": True
    })
    env.reset()
    env_interface = HighwayEnvInterface(env)

    for _ in range(10):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        tboxes, localmap, egostate = env_interface.wrap_observation(obs)
        env.render()

    plt.imshow(env.render())
    plt.show()
