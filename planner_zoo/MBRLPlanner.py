import math
from typing import List, Union
import numpy as np
import torch
import tqdm
import time

from spider.planner_zoo.BasePlanner import BasePlanner

from spider.elements.map import RoutedLocalMap
from spider.elements.trajectory import FrenetTrajectory
from spider.elements.vehicle import VehicleState
from spider.elements.Box import TrackingBoxList, TrackingBox

from spider.sampler.PolynomialSampler import QuinticPolyminalSampler, QuarticPolyminalSampler
from spider.sampler.Combiner import LatLonCombiner
from spider.evaluator import FrenetCostEvaluator

from spider.utils.transform.frenet import FrenetCoordinateTransformer
from spider.utils.collision import BoxCollisionChecker

from spider.RL.agents import MBRLAgent
from spider.RL.reward import FstateControlReward

'''
qzl: 基本的伪代码

state, action, next_state, done = None, None, None, False
_env.reset()
closed_loop = True # 默认闭环，开环的话不会计算奖励函数并储存经验池，纯做policy.act(state)

while True:
    observation = _env.observe()

    ------------------------- Planner内部 ------------------------------
    next_state = Encoder.encode(observation)

    if closed_loop:
        reward, done = RewardFunction.evaluate(state, action, next_state)
        agent.experience_buffer.record(state, action, reward, next_state, done)
        # 注意，当state是none的时候，reward的计算以及经验池的record都是无效的

    if done:
        state, action, next_state = None, None, None
        plan = None
    else:
        state = next_state
        action = agent.policy.act(state)
        plan = Decoder.decode(action)
    ---------------------------------------------------------------------

    if plan is None: _env.reset()
    else: _env.step(plan)
'''


class MBRLPlanner(BasePlanner):
    def __init__(self, config=None, device=None):
        super(MBRLPlanner, self).__init__(config)

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)
            # self.configure(config)

        self.local_map = RoutedLocalMap()
        self.coordinate_transformer = FrenetCoordinateTransformer()  # 要维护几个坐标系呢？
        # self.predictor = None
        self.longitudinal_sampler = QuarticPolyminalSampler(self.config["end_T_candidates"],
                                                            self.config["end_v_candidates"])
        self.lateral_sampler = QuinticPolyminalSampler(self.config["end_s_candidates"],
                                                       self.config["end_l_candidates"])
        self.trajectory_combiner = LatLonCombiner(self.config["steps"], self.config["dt"])  # 默认路径-速度解耦的重新耦合
        self.trajectory_evaluator = FrenetCostEvaluator()
        # self.constraint_checker = None
        self.collision_checker = BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])

        state_dim = self.config["state_veh_num"] * self.config["state_feat_num"]
        action_dim = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.state = None  # ElementFrenetState()
        self.next_state = None
        self.action = None
        self.reward_function = FstateControlReward(self.config)
        self.agent = MBRLAgent(self.config["state_veh_num"], self.config["state_feat_num"], action_dim,
                               self.config["dt"], reward_function=self.reward_function, device=self.device)
        self.closed_loop = self.config["closed_loop"]

        self._candidate_trajectories = None

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "steps": 20,
            "dt": 0.1,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            "lane_width": 3.5,
            "finishing_line": 160,

            "state_veh_num": 10,
            "state_feat_num": 7,
            "closed_loop": True,

            "max_speed": 60 / 3.6,
            "min_speed": 0,
            "max_acceleration": 10,
            "max_deceleration": 10,
            "max_steer" : 35 * math.pi / 180,
            # "max_centripetal_acceleration" : 100,
            "max_curvature": 100,
            "end_s_candidates": (15, 40),
            "end_l_candidates": (-3.5, 0, 3.5),  # s,d采样生成横向轨迹
            "end_v_candidates": tuple(i * 60 / 3.6 / 2 + 1 for i in range(3)),  # 改这一项的时候，要连着限速一起改了
            "end_T_candidates": (3, 5)  # s_dot, T采样生成纵向轨迹
        }

    def get_candidate_trajectories(self):
        return self._candidate_trajectories

    def set_reward_function(self):
        pass

    def set_local_map(self, local_map: RoutedLocalMap):
        self.local_map = local_map

    def build_frenet_lane(self, target_lane_idx):
        if target_lane_idx < 0 or target_lane_idx >= len(self.local_map.lanes):
            raise ValueError("Invalid target lane index")
        target_lane = self.local_map.lanes[target_lane_idx]
        self.coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)

    def match_lanes(self, ego_veh_state: VehicleState):
        # todo:可以放在map的函数里？
        if len(self.local_map.lanes) == 0:
            raise ValueError("No lanes!")

        x, y = ego_veh_state.transform.location.x, ego_veh_state.transform.location.y
        min_idx, min_dist = -1, math.inf
        for idx in range(len(self.local_map.lanes)):
            self.build_frenet_lane(idx)
            fstate = self.coordinate_transformer.cart2frenet(x, y, order=0)
            dist = math.fabs(fstate.l)
            if dist < min_dist:
                min_idx, min_dist = idx, dist
        return min_idx

    def encode_state(self, obstacles: TrackingBoxList, ego_fstate):
        # 元素列表
        # 如果是笛卡尔坐标：(presence,x,y,length,width,heading,speed)
        # 如果是frenet坐标:(presence,s,l,s_dot,l_dot(l_prime),length,width)
        element_list = [torch.tensor([1, ego_fstate.s, ego_fstate.l, ego_fstate.s_dot, ego_fstate.l_prime,
                                      self.config["ego_veh_length"], self.config["ego_veh_width"]])]
        for tb in obstacles:
            if len(element_list) >= self.config["state_veh_num"]:
                break
            x, y, length, width, heading = tb.obb
            v = math.sqrt(tb.vx ** 2 + tb.vy ** 2)
            fstate = self.coordinate_transformer.cart2frenet(x, y, v, heading, order=1)
            element_list.append(torch.tensor([1, fstate.s, fstate.l, fstate.s_dot, fstate.l_prime, length, width]))

        if len(element_list) < self.config["state_veh_num"]:
            add_num = self.config["state_veh_num"] - len(element_list)
            element_list += [torch.tensor([0., 0., 0., 0., 0, 0, 0])] * add_num
        state = torch.cat(element_list).to(torch.float32)  # .unsqueeze(0)
        return state

    def decode_action(self, action, candidate_trajectories):
        optimal_trajectory = candidate_trajectories[action]
        return optimal_trajectory

    def save_experience_buffer(self):
        pass

    def learn(self):
        self.agent.learn()

    def save_model(self, filename: str):
        self.agent.save_transition_model(filename)

    def plan(self, ego_veh_state: VehicleState, obstacles: TrackingBoxList, local_map: RoutedLocalMap = None,
             train=False) \
            -> Union[FrenetTrajectory, None]:
        """
        输入定位、物体、（地图optional,更新频率比较慢。建议在外面单独写set地图的逻辑）
        输出轨迹（FrenetTrajectory）
        """
        t1 = time.time()
        # 储存地图。每条车道代表了一个frenet坐标系，储存地图即储存了lanes，即储存了可供选择的几个frenet坐标系
        if not (local_map is None):
            self.set_local_map(local_map)

        ################################### encode observation into state #################################
        # 把自车位置匹配到对应车道，并且把自车点位转换为Frenet坐标
        # 坐标系建立及坐标转换（车道匹配+车道决策+坐标转换）
        ego_lane_idx = self.match_lanes(ego_veh_state)  # 把自车位置匹配到对应车道
        target_lane_idx = ego_lane_idx  # 目前车道决策还没写上，先默认自车车道，按道理是一个以自车车道输入的函数
        # self.build_frenet_lane(target_lane_idx)
        self.build_frenet_lane(1)
        fstate_start = self.coordinate_transformer.cart2frenet(ego_veh_state.x(), ego_veh_state.y(),
                                                               ego_veh_state.v(), ego_veh_state.yaw(),
                                                               ego_veh_state.a(), ego_veh_state.kappa(), order=2)
        self.next_state = self.encode_state(obstacles, fstate_start)  # 状态编码器
        # todo:这里的state其实应该是笛卡尔坐标下，否则运动学模型会不对
        #################################### data closed loop ####################################

        if self.closed_loop:
            reward, done = self.reward_function.evaluate(self.state, self.action, self.next_state)
            self.agent.record_data(self.state, self.action, reward, self.next_state, done)

            if done:
                self.state, self.next_state, self.action = None, None, None
                return None

        ################################## pick action for state ##################################
        # 轨迹采样
        long_samples = self.longitudinal_sampler.sample((fstate_start.s, fstate_start.s_dot, fstate_start.s_2dot))
        lat_samples = self.lateral_sampler.sample((fstate_start.l, fstate_start.l_prime, fstate_start.l_2prime))
        candidate_trajectories = self.trajectory_combiner.combine(lat_samples, long_samples)
        candidate_trajectories = [self.coordinate_transformer.frenet2cart4traj(t, order=2) for t in candidate_trajectories]

        self.state = self.next_state
        action_sequences = [torch.FloatTensor(traj.convert_to_acc_steer()) for traj in candidate_trajectories]
        egreedy = True if train else False
        action_idx, self.action = self.agent.act(self.state.to(self.device), action_sequences, egreedy=egreedy, epsilon=1.0)  # policy

        ################################### decode action into trajectory #################################

        # 轨迹坐标转换，把每个轨迹点转到笛卡尔坐标
        optimal_trajectory = self.decode_action(action_idx, candidate_trajectories)  # 动作解码器

        self._candidate_trajectories = candidate_trajectories

        t2 = time.time()
        if not train:
            print("Planning Succeed! Time: %.2f seconds, FPS: %.2f" % (t2-t1, 1/(t2-t1)))
            print("action:", self.action)
        return optimal_trajectory


if __name__ == '__main__':
    from spider.elements.map import Lane
    from spider.elements.vehicle import *
    from spider.elements.Box import obb2vertices
    import matplotlib.pyplot as plt


    def test(transition_model_filename=None):
        #################### 输入信息的初始化 ####################

        # 地图信息
        local_map = RoutedLocalMap()
        for idx, yy in enumerate([-3.5, 0, 3.5]):
            xs = np.arange(0, 250.1, 1.0)
            cline = np.column_stack((xs, np.ones_like(xs) * yy))
            lane = Lane(idx, cline, width=3.5, speed_limit=60 / 3.6)
            local_map.lanes.append(lane)
        # 感知信息
        tb_list = TrackingBoxList()
        tb_list.append(TrackingBox(obb=(50, 0, 5, 2, 0), vx=0, vy=0))
        tb_list.append(TrackingBox(obb=(100, 0, 5, 2, 0), vx=0, vy=0))
        # 定位信息
        ego_veh_state = VehicleState(
            transform=Transform(
                location=Location(5., 0, 0),
                rotation=Rotation(0, 0, 0)
            ),
            velocity=Vector3D(0, 0, 0),
            acceleration=Vector3D(0, 0, 0)
        )

        rl_planner = MBRLPlanner({"closed_loop": False})

        if not (transition_model_filename is None):
            rl_planner.agent.load_transition_model(transition_model_filename)
        rl_planner.set_local_map(local_map)

        ################## main loop ########################
        while True:
            if ego_veh_state.x() > 160: break

            # 地图信息更新

            # 感知信息更新
            # obstacles = TrackingBoxList()

            # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
            # ego_veh_state = ...

            traj = rl_planner.plan(ego_veh_state, tb_list, train=False)  # , local_map)

            if traj is None:
                break

            # 可视化
            plt.cla()
            for lane in local_map.lanes:
                plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图
            vertices = obb2vertices([ego_veh_state.x(), ego_veh_state.y(), 5., 2., ego_veh_state.yaw()])
            vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
            plt.plot(vertices[:, 0], vertices[:, 1], color='blue', linestyle='-', linewidth=1.5)  # 画自车
            for tb in tb_list:
                vertices = tb.vertices
                vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
                plt.plot(vertices[:, 0], vertices[:, 1], color='black', linestyle='-', linewidth=1.5)  # 画他车
            plt.plot(traj.x, traj.y, 'r-.', lw=1)  # 画轨迹
            plt.axis('equal')
            plt.xlim([ego_veh_state.x() - 5, ego_veh_state.x() + 100])
            plt.ylim([ego_veh_state.y() - 5, ego_veh_state.y() + 5])
            plt.pause(0.01)

            # 控制+定位，假设完美控制到下一个轨迹点
            ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
                = traj.x[1], traj.y[1], traj.heading[1]
            ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
                = traj.v[1], traj.a[1], traj.curvature[1]
        pass


    def train(episodes=200, transition_model_filename="./model.pth", resume: str = None):
        #################### 输入信息的初始化 ####################

        # 地图信息
        local_map = RoutedLocalMap()
        for idx, yy in enumerate([-3.5, 0, 3.5]):
            xs = np.arange(0, 250.1, 1.0)
            cline = np.column_stack((xs, np.ones_like(xs) * yy))
            lane = Lane(idx, cline, width=3.5, speed_limit=60 / 3.6)
            local_map.lanes.append(lane)
        # 感知信息
        tb_list = TrackingBoxList()
        tb_list.append(TrackingBox(obb=(50, 0, 5, 2, 0), vx=0, vy=0))
        tb_list.append(TrackingBox(obb=(100, 0, 5, 2, 0), vx=0, vy=0))
        # 定位信息
        # ego_veh_state = VehicleState(
        #     transform=Transform(
        #         location=Location(5., 0, 0),
        #         rotation=Rotation(0, 0, 0)
        #     ),
        #     velocity=Vector3D(0, 0, 0),
        #     acceleration=Vector3D(0, 0, 0)
        # )

        rl_planner = MBRLPlanner()
        rl_planner.set_local_map(local_map)
        if not (resume is None):
            rl_planner.agent.load_transition_model(resume)

        ################## main loop ########################
        for episode in tqdm.tqdm(range(episodes)):
            ego_veh_state = VehicleState(
                transform=Transform(
                    location=Location(5., 0, 0),
                    rotation=Rotation(0, 0, 0)
                ),
                velocity=Vector3D(0, 0, 0),
                acceleration=Vector3D(0, 0, 0)
            )

            while True:
                # if ego_veh_state.x() > 160: break
                # 地图信息更新

                # 感知信息更新
                # obstacles = TrackingBoxList()

                # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
                # ego_veh_state = ...

                traj = rl_planner.plan(ego_veh_state, tb_list, train=True)  # , local_map)
                rl_planner.learn() # todo: qzl: 环境模型是不是应该在数据都采完之后统一学习？？

                if traj is None:  # 表示done，需要reset
                    break

                # 控制+定位，假设完美控制到下一个轨迹点
                ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
                    = traj.x[1], traj.y[1], traj.heading[1]
                ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
                    = traj.v[1], traj.a[1], traj.curvature[1]

        ################## save model ########################
        rl_planner.save_model(transition_model_filename)


    # train(episodes=200, transition_model_filename='model_for_mbrl.pth')  # ,resume='./model_for_mfrl.pth')
    test('model_for_mbrl.pth') #'./model_for_mfrl.pth'
