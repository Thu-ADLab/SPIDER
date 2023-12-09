import math
from typing import List, Union
import numpy as np
import torch
import tqdm

from spider.planner_zoo.BasicPlanner import BasicPlanner

from spider.elements.map import RoutedLocalMap
from spider.elements.trajectory import FrenetTrajectory
from spider.elements.vehicle import VehicleState
from spider.elements.Box import TrackingBoxList, TrackingBox

from spider.sampler.PolynomialSampler import QuinticPolyminalSampler, QuarticPolyminalSampler
from spider.sampler.Combiner import LatLonCombiner
from spider.evaluator import FrenetCostEvaluator

from spider.utils.transform.frenet import FrenetCoordinateTransformer
from spider.utils.collision import BoxCollisionChecker

from spider.RL.agents import DQNAgent
from spider.RL.state import ElementFrenetState
from spider.RL.reward import FstateTrajectoryReward

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


class MFRLPlanner(BasicPlanner):
    def __init__(self, config=None, device=None):
        super(MFRLPlanner, self).__init__()

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)
            # self.configure(config)

        self.local_map = RoutedLocalMap()
        self.coordinate_transformer = FrenetCoordinateTransformer() # 要维护几个坐标系呢？
        # self.predictor = None
        self.longitudinal_sampler = QuarticPolyminalSampler(self.config["end_T_candidates"],
                                                            self.config["end_v_candidates"])
        self.lateral_sampler = QuinticPolyminalSampler(self.config["end_s_candidates"],
                                                       self.config["end_l_candidates"])
        self.trajectory_combiner = LatLonCombiner(self.config["steps"], self.config["dt"]) # 默认路径-速度解耦的重新耦合
        self.trajectory_evaluator = FrenetCostEvaluator()
        # self.constraint_checker = None
        self.collision_checker = BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])

        state_dim = self.config["state_veh_num"] * self.config["state_feat_num"]
        action_dim = len(self.config["end_s_candidates"]) * len(self.config["end_l_candidates"]) * \
                     len(self.config["end_T_candidates"]) * len(self.config["end_v_candidates"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.state = None#ElementFrenetState()
        self.next_state = None
        self.action = None
        self.agent = DQNAgent(state_dim, action_dim, gamma=self.config["discount_factor"], device=self.device)
        self.reward_function = FstateTrajectoryReward(self.config)
        self.closed_loop = self.config["closed_loop"]


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "steps": 50,
            "dt": 0.1,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            "lane_width": 3.5,
            "finishing_line": 160,


            "state_veh_num": 10,
            "state_feat_num": 7,
            "discount_factor": 0.95,
            "closed_loop": True,


            "max_speed": 60/3.6,
            "min_speed": 0,
            "max_acceleration": 10,
            "max_deceleration": 10,
            # "max_centripetal_acceleration" : 100,
            "max_curvature": 100,
            "end_s_candidates": (15,40),
            "end_l_candidates": (-3.5,0,3.5), # s,d采样生成横向轨迹
            "end_v_candidates": tuple(i*60/3.6/2 + 1 for i in range(3)), # 改这一项的时候，要连着限速一起改了
            "end_T_candidates": (3,5) # s_dot, T采样生成纵向轨迹
        }

    def configure(self, config: dict):
        self.__init__(config)

    def set_reward_function(self):
        pass


    def set_local_map(self, local_map:RoutedLocalMap):
        self.local_map = local_map

    def build_frenet_lane(self, target_lane_idx):
        if target_lane_idx < 0 or target_lane_idx >= len(self.local_map.lanes):
            raise ValueError("Invalid target lane index")
        target_lane = self.local_map.lanes[target_lane_idx]
        self.coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)

    def match_lanes(self, ego_veh_state:VehicleState):
        # todo:可以放在map的函数里？
        if len(self.local_map.lanes) == 0:
            raise ValueError("No lanes!")

        x,y = ego_veh_state.transform.location.x, ego_veh_state.transform.location.y
        min_idx, min_dist = -1, math.inf
        for idx in range(len(self.local_map.lanes)):
            self.build_frenet_lane(idx)
            fstate = self.coordinate_transformer.cart2frenet(x, y, order=0)
            dist = math.fabs(fstate.l)
            if dist<min_dist:
                min_idx, min_dist = idx, dist
        return min_idx

    def encode_state(self, obstacles:TrackingBoxList, ego_fstate):
        # todo:这里state没有归一化。这么说来，还是需要一个StateEncoder来储存一下所有这些方法（包括编码和归一化）
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
            element_list.append(torch.tensor([1,fstate.s, fstate.l, fstate.s_dot, fstate.l_prime, length, width]))

        if len(element_list) < self.config["state_veh_num"]:
            add_num = self.config["state_veh_num"] - len(element_list)
            element_list += [torch.tensor([0.,0.,0.,0.,0,0,0])] * add_num
        state = torch.cat(element_list).to(torch.float32)#.unsqueeze(0)
        return state

    def decode_action(self, action, candidate_trajectories):
        optimal_trajectory = candidate_trajectories[action]
        return optimal_trajectory

    def save_experience_buffer(self):
        pass

    def learn(self):
        self.agent.learn()

    def save_model(self, filename:str):
        self.agent.save_q_network_model(filename)

    def plan(self, ego_veh_state:VehicleState, obstacles:TrackingBoxList, local_map:RoutedLocalMap=None, train=False) \
            -> Union[FrenetTrajectory, None]:
        """
        输入定位、物体、（地图optional,更新频率比较慢。建议在外面单独写set地图的逻辑）
        输出轨迹（FrenetTrajectory）
        """
        # 储存地图。每条车道代表了一个frenet坐标系，储存地图即储存了lanes，即储存了可供选择的几个frenet坐标系
        if not (local_map is None):
            self.set_local_map(local_map)

        ################################### encode observation into state #################################
        # 把自车位置匹配到对应车道，并且把自车点位转换为Frenet坐标
        # 坐标系建立及坐标转换（车道匹配+车道决策+坐标转换）
        # ego_lane_idx = self.match_lanes(ego_veh_state)  # 把自车位置匹配到对应车道
        # target_lane_idx = ego_lane_idx  # 目前车道决策还没写上，先默认自车车道，按道理是一个以自车车道输入的函数
        # self.build_frenet_lane(target_lane_idx)
        self.build_frenet_lane(1)
        fstate_start = self.coordinate_transformer.cart2frenet(ego_veh_state.x(), ego_veh_state.y(),
                                                               ego_veh_state.v(), ego_veh_state.yaw(),
                                                               ego_veh_state.a(), ego_veh_state.kappa(), order=2)
        self.next_state = self.encode_state(obstacles, fstate_start)  # 状态编码器

        #################################### data closed loop ####################################

        if self.closed_loop:
            reward, done = self.reward_function.evaluate(self.state, self.action, self.next_state)
            self.agent.record_data(self.state, self.action, reward, self.next_state, done)

            if done:
                self.state,self.next_state,self.action = None, None, None
                return None


        ################################## pick action for state ##################################

        self.state = self.next_state
        egreedy = True if train else False
        self.action = self.agent.act(self.state.to(self.device),egreedy=egreedy)  # policy

        ################################### decode action into trajectory #################################
        # 轨迹采样
        long_samples = self.longitudinal_sampler.sample((fstate_start.s, fstate_start.s_dot, fstate_start.s_2dot))
        lat_samples = self.lateral_sampler.sample((fstate_start.l, fstate_start.l_prime, fstate_start.l_2prime))
        candidate_trajectories = self.trajectory_combiner.combine(lat_samples, long_samples)
        self.reward_function.set_trajectory_candidates(candidate_trajectories)

        # 轨迹坐标转换，把每个轨迹点转到笛卡尔坐标
        # todo:qzl: 其实candidates_trajectories不用算出来的，会耗损计算资源，能不能储存Generator形式的？调用的时候再进行计算
        candidate_trajectories = [self.coordinate_transformer.frenet2cart4traj(t, order=2) for t in candidate_trajectories]
        optimal_trajectory = self.decode_action(self.action,candidate_trajectories)  # 动作解码器

        return optimal_trajectory




if __name__ == '__main__':
    from spider.elements.map import Lane
    from spider.elements.vehicle import *
    from spider.elements.Box import obb2vertices
    import matplotlib.pyplot as plt
    import cv2


    def test(q_network_model_filename=None, save_video=False):
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

        rl_planner = MFRLPlanner()
        rl_planner.configure({"closed_loop": False})

        if not (q_network_model_filename is None):
            rl_planner.agent.load_q_network_model(q_network_model_filename)
        rl_planner.set_local_map(local_map)

        if save_video:
            # 设置视频帧数和时长
            frame_rate = 15
            # 创建视频编写器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') #'XVID'
            video_name = 'test_mfrl_50000.mp4'
            video_writer = None

        ################## main loop ########################
        while True:
            if ego_veh_state.x() > 160: break

            # 地图信息更新

            # 感知信息更新
            # tb_list = TrackingBoxList()

            # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
            # ego_veh_state = ...

            traj = rl_planner.plan(ego_veh_state, tb_list, train=False)#, local_map)

            if traj is None:
                break

            # 可视化
            plt.cla()
            for lane in local_map.lanes:
                plt.plot(lane.centerline[:,0], lane.centerline[:,1], color='gray', linestyle='--', lw= 1.5) # 画地图
            vertices = obb2vertices([ego_veh_state.x(), ego_veh_state.y(),5.,2.,ego_veh_state.yaw()])
            vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
            plt.plot(vertices[:, 0], vertices[:, 1], color='blue', linestyle='-', linewidth=1.5) # 画自车
            for tb in tb_list:
                vertices = tb.vertices
                vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
                plt.plot(vertices[:, 0], vertices[:, 1], color='black', linestyle='-', linewidth=1.5)  # 画他车
            plt.plot(traj.x,traj.y,'r-.',lw=1) # 画轨迹
            plt.axis('equal')
            plt.xlim([ego_veh_state.x()-5, ego_veh_state.x()+100])
            plt.ylim([ego_veh_state.y()-5,ego_veh_state.y()+5])
            plt.pause(0.01)
            # 将Matplotlib图形保存为图像
            if save_video:
                fig = plt.gcf()
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(np.append(np.flip(fig.get_size_inches()) * fig.dpi, 3).astype(np.int))
                if video_writer is None:
                    video_writer = cv2.VideoWriter(video_name, fourcc, frame_rate, (image.shape[1],image.shape[0]))
                video_writer.write(image)

            # 控制+定位，假设完美控制到下一个轨迹点
            ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
                = traj.x[1], traj.y[1], traj.heading[1]
            ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
                = traj.v[1], traj.a[1], traj.curvature[1]

        if save_video:
            video_writer.release()
            print("video_writer.release()")


    def train(episodes=200, q_network_model_filename="./model.pth", resume: str = None):
        # todo:以后可以建个Trainer类，用来训
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

        rl_planner = MFRLPlanner()
        rl_planner.set_local_map(local_map)
        if not (resume is None):
            rl_planner.agent.load_q_network_model(resume)

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
                # tb_list = TrackingBoxList()

                # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
                # ego_veh_state = ...

                traj = rl_planner.plan(ego_veh_state, tb_list, train=True)  # , local_map)
                rl_planner.learn()

                if traj is None: # 表示done，需要reset
                    break

                # 控制+定位，假设完美控制到下一个轨迹点
                ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
                    = traj.x[1], traj.y[1], traj.heading[1]
                ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
                    = traj.v[1], traj.a[1], traj.curvature[1]

        ################## save model ########################
        rl_planner.save_model(q_network_model_filename)

    # train(episodes=200, q_network_model_filename='./model_for_mfrl.pth')#,resume='./model_for_mfrl.pth')
    test('model_for_mfrl.pth',save_video=True)#'model_for_mfrl.pth'

    # def test_intersection(q_network_model_filename=None, save_video=False):
    #     #################### 输入信息的初始化 ####################
    #
    #     # 地图信息
    #     local_map = RoutedLocalMap()
    #     ys = np.arange(-40, -10, 1.0)
    #     cline1 = np.column_stack((np.ones_like(ys) * 3.5 / 2, ys))
    #     xs = np.arange(-11, -60, -1.0)
    #     cline3 = np.column_stack((xs, np.ones_like(xs) * 3.5 / 2))
    #     cline2 = []
    #     length = 10 * math.pi * 0.5
    #     sample_num_for_cline2 = int(math.ceil(length / 1) + 1)
    #     thetas = np.linspace(0, math.pi / 2, sample_num_for_cline2)  # 圆心角
    #     radius = 10 + 3.5 / 2
    #     for theta in thetas:
    #         x = -10 + radius * math.cos(theta)
    #         y = -10 + radius * math.sin(theta)
    #         cline2.append([x, y])
    #     cline2 = np.array(cline2)
    #     cline = np.vstack((cline1, cline2, cline3))
    #     lane = Lane(0, cline, width=3.5, speed_limit=30 / 3.6)
    #     local_map.lanes.append(lane)
    #     # 感知信息
    #     tb_list = TrackingBoxList()
    #     tb_list.append(TrackingBox(obb=(-3.5 / 2, 20, 5, 2, -math.pi / 2), vx=0, vy=-5))
    #
    #     # 定位信息
    #     ego_veh_state = VehicleState(
    #         transform=Transform(
    #             location=Location(3.5 / 2., -29.0, 0),
    #             rotation=Rotation(0, math.pi / 2, 0)
    #         ),
    #         velocity=Vector3D(0, 0, 0),
    #         acceleration=Vector3D(0, 0, 0)
    #     )
    #
    #     rl_planner = MFRLPlanner()
    #     rl_planner.configure({
    #         "closed_loop": False,
    #
    #         "end_l_candidates": (-0.8, 0, 0.8),
    #         "steps": 20,
    #         "max_speed": 30 / 3.6,
    #         "end_s_candidates": (5, 10, 20, 30),
    #         "end_v_candidates": tuple(i * 30 / 3.6 / 3 for i in range(4)),
    #         "end_T_candidates": (0.01, 1, 2, 4),
    #     })
    #
    #     if not (q_network_model_filename is None):
    #         rl_planner.agent.load_q_network_model(q_network_model_filename)
    #     rl_planner.set_local_map(local_map)
    #
    #     if save_video:
    #         # 设置视频帧数和时长
    #         frame_rate = 15
    #         # 创建视频编写器
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v') #'XVID'
    #         video_name = 'test_mfrl_intersection_0.mp4'
    #         video_writer = None
    #     plt.figure(figsize=(6,6))
    #     ################## main loop ########################
    #     while True:
    #         if ego_veh_state.x() < -20: break
    #
    #         # 地图信息更新
    #
    #         # 感知信息更新
    #         for i in range(len(tb_list)):
    #             tb = tb_list[i]
    #             x, y, l, w, h = tb.obb
    #             x += tb.vx * rl_planner.config["dt"]
    #             y += tb.vy * rl_planner.config["dt"]
    #             tb_list[i].setObb((x, y, l, w, h))
    #
    #         # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
    #         # ego_veh_state = ...
    #
    #         traj = rl_planner.plan(ego_veh_state, tb_list, train=False)#, local_map)
    #
    #         if traj is None:
    #             break
    #
    #         # 可视化
    #         plt.cla()
    #         for lane in local_map.lanes:
    #             plt.plot(lane.centerline[:,0], lane.centerline[:,1], color='gray', linestyle='--', lw= 1.5) # 画地图
    #         vertices = obb2vertices([ego_veh_state.x(), ego_veh_state.y(),5.,2.,ego_veh_state.yaw()])
    #         vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    #         plt.plot(vertices[:, 0], vertices[:, 1], color='blue', linestyle='-', linewidth=1.5) # 画自车
    #         for tb in tb_list:
    #             vertices = tb.vertices
    #             vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    #             plt.plot(vertices[:, 0], vertices[:, 1], color='black', linestyle='-', linewidth=1.5)  # 画他车
    #         plt.plot(traj.x,traj.y,'r-.',lw=1) # 画轨迹
    #         plt.axis('equal')
    #         plt.xlim([-25, 25])
    #         plt.ylim([-30, 30])
    #         plt.pause(0.01)
    #         # 将Matplotlib图形保存为图像
    #         if save_video:
    #             fig = plt.gcf()
    #             fig.canvas.draw()
    #             image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             image = image.reshape(np.append(np.flip(fig.get_size_inches()) * fig.dpi, 3).astype(np.int))
    #             if video_writer is None:
    #                 video_writer = cv2.VideoWriter(video_name, fourcc, frame_rate, (image.shape[1],image.shape[0]))
    #             video_writer.write(image)
    #
    #         # 控制+定位，假设完美控制到下一个轨迹点
    #         ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
    #             = traj.x[1], traj.y[1], traj.heading[1]
    #         ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
    #             = traj.v[1], traj.a[1], traj.curvature[1]
    #
    #     if save_video:
    #         video_writer.release()
    #         print("video_writer.release()")
    #
    #
    # def train_intersection(episodes=200, q_network_model_filename="./model.pth", resume: str = None):
    #     # todo:以后可以建个Trainer类，用来训
    #     #################### 输入信息的初始化 ####################
    #
    #     # 地图信息
    #     local_map = RoutedLocalMap()
    #     ys = np.arange(-40, -10, 1.0)
    #     cline1 = np.column_stack((np.ones_like(ys) * 3.5 / 2, ys))
    #     xs = np.arange(-11, -60, -1.0)
    #     cline3 = np.column_stack((xs, np.ones_like(xs) * 3.5 / 2))
    #     cline2 = []
    #     length = 10 * math.pi * 0.5
    #     sample_num_for_cline2 = int(math.ceil(length / 1) + 1)
    #     thetas = np.linspace(0, math.pi / 2, sample_num_for_cline2)  # 圆心角
    #     radius = 10 + 3.5 / 2
    #     for theta in thetas:
    #         x = -10 + radius * math.cos(theta)
    #         y = -10 + radius * math.sin(theta)
    #         cline2.append([x, y])
    #     cline2 = np.array(cline2)
    #     cline = np.vstack((cline1, cline2, cline3))
    #     lane = Lane(0, cline, width=3.5, speed_limit=30 / 3.6)
    #     local_map.lanes.append(lane)
    #     # 感知信息
    #     tb_list = TrackingBoxList()
    #     tb_list.append(TrackingBox(obb=(-3.5 / 2, 20, 5, 2, -math.pi / 2), vx=0, vy=-5))
    #     # 定位信息
    #     # ego_veh_state = VehicleState(
    #     #     transform=Transform(
    #     #         location=Location(5., 0, 0),
    #     #         rotation=Rotation(0, 0, 0)
    #     #     ),
    #     #     velocity=Vector3D(0, 0, 0),
    #     #     acceleration=Vector3D(0, 0, 0)
    #     # )
    #
    #     rl_planner = MFRLPlanner()
    #     rl_planner.configure({
    #         "end_l_candidates": (-0.8, 0, 0.8),
    #         "steps": 20,
    #         "max_speed": 30 / 3.6,
    #         "end_s_candidates": (5, 10, 20, 30),
    #         "end_v_candidates": tuple(i * 30 / 3.6 / 3 for i in range(4)),
    #         "end_T_candidates": (0.01, 1, 2, 4),
    #     })
    #     rl_planner.set_local_map(local_map)
    #     if not (resume is None):
    #         rl_planner.agent.load_q_network_model(resume)
    #
    #     ################## main loop ########################
    #     for episode in tqdm.tqdm(range(episodes)):
    #         ego_veh_state = VehicleState(
    #             transform=Transform(
    #                 location=Location(3.5 / 2., -29.0, 0),
    #                 rotation=Rotation(0, math.pi / 2, 0)
    #             ),
    #             velocity=Vector3D(0, 0, 0),
    #             acceleration=Vector3D(0, 0, 0)
    #         )
    #
    #         while True:
    #             # if ego_veh_state.x() > 160: break
    #             # 地图信息更新
    #
    #             # 感知信息更新
    #             for i in range(len(tb_list)):
    #                 tb = tb_list[i]
    #                 x, y, l, w, h = tb.obb
    #                 x += tb.vx * rl_planner.config["dt"]
    #                 y += tb.vy * rl_planner.config["dt"]
    #                 tb_list[i].setObb((x, y, l, w, h))
    #
    #             # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
    #             # ego_veh_state = ...
    #
    #             traj = rl_planner.plan(ego_veh_state, tb_list, train=True)  # , local_map)
    #             rl_planner.learn()
    #
    #             if traj is None: # 表示done，需要reset
    #                 break
    #
    #             # 控制+定位，假设完美控制到下一个轨迹点
    #             ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
    #                 = traj.x[1], traj.y[1], traj.heading[1]
    #             ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
    #                 = traj.v[1], traj.a[1], traj.curvature[1]
    #
    #     ################## save model ########################
    #     rl_planner.save_model(q_network_model_filename)
    #
    # test_intersection()