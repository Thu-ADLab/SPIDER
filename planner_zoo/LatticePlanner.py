import math
from typing import List
import time
import numpy as np

import spider
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

from spider.constraints import CartConstriantChecker


class LatticePlanner(BasicPlanner):
    def __init__(self, config=None):
        super(LatticePlanner, self).__init__()

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

        # self.collision_checker = BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])
        self.constraint_checker = CartConstriantChecker(
            self.config, BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])
        )

        self._candidate_trajectories = None
        self._candidate_trajectories_cost = None


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 50,
            "dt": 0.1,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            "max_speed": 60/3.6,
            "min_speed": 0,
            "max_acceleration": 10,
            "max_deceleration": 10,
            # "max_centripetal_acceleration" : 100,
            "max_curvature": 100,
            "end_s_candidates": (10,20,40,60),
            "end_l_candidates": (-0.8,0,0.8), # s,d采样生成横向轨迹 (-3.5, 0, 3.5), #
            "end_v_candidates": tuple(i*60/3.6/3 for i in range(4)), # 改这一项的时候，要连着限速一起改了
            "end_T_candidates": (1,2,4,8), # s_dot, T采样生成纵向轨迹

            "constraint_flags": {
                spider.CONSTRIANT_SPEED_UB,
                spider.CONSTRIANT_SPEED_LB,
                spider.CONSTRIANT_ACCELERATION,
                spider.CONSTRIANT_DECELERATION,
                spider.CONSTRIANT_CURVATURE
            },
        }

    def configure(self, config: dict):
        self.__init__(config)
        # if config:
            # self.config.update(config)
        # self.speed_bound = [self.config["min_speed"], self.config["max_speed"]]
        # self.acc_bound = [-self.config["max_deceleration"], self.config["max_acceleration"]]
        # self.max_curvature = self.config["max_curvature"]
        # self.end_s_candidates = self.config["end_s_candidates"]
        # self.end_d_candidates = self.config["end_d_candidates"]
        # self.end_v_candidates = self.config["end_v_candidates"]
        # self.end_T_candidates = self.config["end_T_candidates"]


    def get_candidate_traj_with_cost(self):
        return self._candidate_trajectories, self._candidate_trajectories_cost


    def constraint_check(self, sorted_candidate_trajectories:List[FrenetTrajectory], sorted_cost, obstacles:TrackingBoxList):
        for traj, cost in zip(sorted_candidate_trajectories, sorted_cost):
            if self.constraint_checker.check(traj, obstacles):
                return traj, cost

            # if np.any(np.array(traj.v) > self.config["max_speed"]): continue
            # if np.any(np.array(traj.v) < self.config["min_speed"]): continue
            # if np.any(np.array(traj.a) > self.config["max_acceleration"]): continue
            # if np.any(np.array(traj.a) < -self.config["max_deceleration"]): continue
            # if np.any(np.abs(traj.curvature) > self.config["max_curvature"]): continue
            # if self.collision_checker.check_trajectory(traj, obstacles): continue
            # return traj, cost


        return None, 0

    def set_local_map(self, local_map:RoutedLocalMap):
        self.local_map = local_map

    def build_frenet_lane(self, target_lane_idx):
        if target_lane_idx < 0 or target_lane_idx >= len(self.local_map.lanes):
            raise ValueError("Invalid target lane index")
        target_lane = self.local_map.lanes[target_lane_idx]
        self.coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)

    # def match_lanes(self, ego_veh_state:VehicleState):
    #     # 已经放在map的函数里了
    #     if len(self.local_map.lanes) == 0:
    #         raise ValueError("No lanes!")
    #
    #     x,y = ego_veh_state.x(), ego_veh_state.y()
    #     min_idx, min_dist = -1, math.inf
    #     for idx in range(len(self.local_map.lanes)):
    #         self.build_frenet_lane(idx)
    #         fstate = self.coordinate_transformer.cart2frenet(x, y, order=0)
    #         dist = math.fabs(fstate.l)
    #         if dist<min_dist:
    #             min_idx, min_dist = idx, dist
    #     return min_idx


    def plan(self, ego_veh_state:VehicleState, obstacles:TrackingBoxList, local_map:RoutedLocalMap=None) -> FrenetTrajectory:
        """
        输入定位、物体、（地图optional,更新频率比较慢。建议在外面单独写set地图的逻辑）
        输出轨迹（FrenetTrajectory）
        """
        t1 = time.time()
        # 储存地图。每条车道代表了一个frenet坐标系，储存地图即储存了lanes，即储存了可供选择的几个frenet坐标系
        if not (local_map is None):
            self.set_local_map(local_map)

        # 把自车位置匹配到对应车道，并且把自车点位转换为Frenet坐标
        # 坐标系建立及坐标转换（车道匹配+车道决策+坐标转换）
        ego_lane_idx = self.local_map.match_lane(ego_veh_state)#self.match_lanes(ego_veh_state)  # 把自车位置匹配到对应车道
        # todo: 加上车道决策的部分
        target_lane_idx = ego_lane_idx  # 目前车道决策还没写上，先默认自车车道，按道理是一个以自车车道输入的函数
        self.build_frenet_lane(target_lane_idx)
        fstate_start = self.coordinate_transformer.cart2frenet(ego_veh_state.x(), ego_veh_state.y(),
                                                               ego_veh_state.v(), ego_veh_state.yaw(),
                                                               ego_veh_state.a(), ego_veh_state.kappa(), order=2)

        # 预测
        predicted_obstacles = obstacles.predict(self.config["dt"] * np.arange(self.config["steps"]))

        # 轨迹采样
        long_samples = self.longitudinal_sampler.sample((fstate_start.s, fstate_start.s_dot, fstate_start.s_2dot))
        lat_samples = self.lateral_sampler.sample((fstate_start.l, fstate_start.l_prime, fstate_start.l_2prime))
        candidate_trajectories = self.trajectory_combiner.combine(lat_samples, long_samples)

        # 轨迹坐标转换，把每个轨迹点转到笛卡尔坐标
        candidate_trajectories = [self.coordinate_transformer.frenet2cart4traj(t, order=2) for t in candidate_trajectories]

        # 评估+筛选
        sorted_candidates, sorted_cost = self.trajectory_evaluator.evaluate_candidates(candidate_trajectories)
        optimal_trajectory, min_cost = self.constraint_check(sorted_candidates, sorted_cost, predicted_obstacles)

        self._candidate_trajectories, self._candidate_trajectories_cost = sorted_candidates, sorted_cost

        if not (optimal_trajectory is None):
            print("Optimal trajectory found! s_dot_end=%.2f,l_end=%.2f" %
                  (optimal_trajectory.s_dot[-1], optimal_trajectory.l[-1]))
        else:
            print("WARNING: NO feasible trajectory!")

        t2 = time.time()
        print("Planning Succeed! Time: %.2f seconds, FPS: %.2f" % (t2 - t1, 1 / (t2 - t1)))

        return optimal_trajectory


if __name__ == '__main__':
    from spider.elements.map import Lane
    from spider.elements.vehicle import *
    from spider.elements.Box import obb2vertices
    import matplotlib.pyplot as plt
    import cv2


    def multilane():

        #################### 输入信息的初始化 ####################
        # 定位信息
        ego_veh_state = VehicleState(
            transform=Transform(
                location=Location(5.,0.01,0),
                rotation=Rotation(0,0,0)
            ),
            velocity=Vector3D(0,0,0),
            acceleration=Vector3D(0,0,0)
        )
        # 地图信息
        local_map = RoutedLocalMap()
        for idx,yy in enumerate([-3.5,0,3.5]):
            xs = np.arange(0,250.1,1.0)
            cline = np.column_stack((xs, np.ones_like(xs)*yy))
            lane = Lane(idx, cline, width=3.5, speed_limit=60/3.6)
            local_map.lanes.append(lane)
        # 感知信息
        tb_list = TrackingBoxList()
        tb_list.append(TrackingBox(obb=(50, 0, 5, 2, 0), vx=0, vy=0))
        tb_list.append(TrackingBox(obb=(100, 0, 5, 2, 0), vx=0, vy=0))

        lattice_planner = LatticePlanner()
        lattice_planner.configure({"end_l_candidates": (-3.5, 0, 3.5)})
        lattice_planner.set_local_map(local_map)

        save_video = True
        if save_video:
            # 设置视频帧数和时长
            frame_rate = 15
            # 创建视频编写器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID'
            video_name = 'test_lattice_multilane.mp4'
            video_writer = None

        ################## main loop ########################
        while True:
            if ego_veh_state.x() > 160: break

            # 地图信息更新

            # 感知信息更新，这里假设完美感知+其他车全部静止


            # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
            # ego_veh_state = ...

            traj = lattice_planner.plan(ego_veh_state, tb_list)#, local_map)

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
                    video_writer = cv2.VideoWriter(video_name, fourcc, frame_rate, (image.shape[1], image.shape[0]))
                video_writer.write(image)

            # 控制+定位，假设完美控制到下一个轨迹点
            ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
                = traj.x[1], traj.y[1], traj.heading[1]
            ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
                = traj.v[1], traj.a[1], traj.curvature[1]

        if save_video:
            video_writer.release()
            print("video_writer.release()")


    def intersection():


        #################### 输入信息的初始化 ####################
        # 定位信息
        ego_veh_state = VehicleState(
            transform=Transform(
                location=Location(3.5/2., -29.0, 0),
                rotation=Rotation(0, math.pi/2, 0)
            ),
            velocity=Vector3D(0, 0, 0),
            acceleration=Vector3D(0, 0, 0)
        )
        # 地图信息
        local_map = RoutedLocalMap()
        ys = np.arange(-40, -10, 1.0)
        cline1 = np.column_stack((np.ones_like(ys)* 3.5/2, ys))
        xs = np.arange(-11, -60, -1.0)
        cline3 = np.column_stack((xs, np.ones_like(xs) * 3.5/2))
        cline2 = []
        length = 10 * math.pi * 0.5
        sample_num_for_cline2 = int(math.ceil(length/1) + 1)
        thetas = np.linspace(0, math.pi/2, sample_num_for_cline2) # 圆心角
        radius = 10+3.5/2
        for theta in thetas: # qzl:注意，这里给的是一个圆弧直接连接，但实际上应该保证曲率连续，选用螺旋线，类似高速公路缓和曲线
            x = -10 + radius * math.cos(theta)
            y = -10 + radius * math.sin(theta)
            cline2.append([x,y])
        cline2 = np.array(cline2)
        cline = np.vstack((cline1,cline2,cline3))
        lane = Lane(0, cline, width=3.5, speed_limit=30 / 3.6)
        local_map.lanes.append(lane)
        # 感知信息
        tb_list = TrackingBoxList()
        tb_list.append(TrackingBox(obb=(-3.5/2, 20, 5, 2, -math.pi/2), vx=0, vy=-5))

        lattice_planner = LatticePlanner()
        lattice_planner.configure({
            "end_l_candidates": (-0.8, 0, 0.8),
            "steps": 20,
            "max_speed": 30/3.6,
            "end_s_candidates":(5,10,20,30),
            "end_v_candidates":tuple(i*30/3.6/3 for i in range(4)),
            "end_T_candidates": (0.01, 1,2, 4),
        })
        lattice_planner.set_local_map(local_map)

        save_video = True
        if save_video:
            # 设置视频帧数和时长
            frame_rate = 15
            # 创建视频编写器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID'
            video_name = 'test_lattice_intersection.mp4'
            video_writer = None
        plt.figure(figsize=(6, 6))
        ################## main loop ########################
        while True:
            if ego_veh_state.x() < -20: break

            # 地图信息更新

            # 感知信息更新，这里假设完美感知+其他车全部静止
            for i in range(len(tb_list)):
                tb = tb_list[i]
                x,y,l,w,h = tb.obb
                x += tb.vx * lattice_planner.config["dt"]
                y += tb.vy * lattice_planner.config["dt"]
                tb_list[i].setObb((x,y,l,w,h))

            # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
            # ego_veh_state = ...

            traj = lattice_planner.plan(ego_veh_state, tb_list)  # , local_map)

            # 可视化
            plt.cla()
            for lane in local_map.lanes:
                plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1)  # 画地图
            #-------------------画地图
            line1 = cline1.copy()
            for delta_x in np.array([0.5,-0.5,-1.5])*3.5:
                plt.plot(line1[:, 0] + delta_x, line1[:, 1], color='black', linestyle='-', lw=1.5)

            line2 = cline3.copy()
            for delta_y in np.array([0.5, -0.5, -1.5]) * 3.5:
                plt.plot(line2[:, 0], line2[:, 1] + delta_y, color='black', linestyle='-', lw=1.5)

            line3 = line1.copy()
            line3[:,1] *= -1
            for delta_x in np.array([0.5, -0.5, -1.5]) * 3.5:
                plt.plot(line3[:, 0] + delta_x, line3[:, 1], color='black', linestyle='-', lw=1.5)

            line4 = line2.copy()
            line4[:, 0] *= -1
            for delta_y in np.array([0.5, -0.5, -1.5]) * 3.5:
                plt.plot(line4[:, 0], line4[:, 1] + delta_y, color='black', linestyle='-', lw=1.5)
            #-------------------
            vertices = obb2vertices([ego_veh_state.x(), ego_veh_state.y(), 5., 2., ego_veh_state.yaw()])
            vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
            plt.plot(vertices[:, 0], vertices[:, 1], color='blue', linestyle='-', linewidth=1.5)  # 画自车
            for tb in tb_list:
                vertices = tb.vertices
                vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
                plt.plot(vertices[:, 0], vertices[:, 1], color='black', linestyle='-', linewidth=1.5)  # 画他车
            plt.plot(traj.x, traj.y, 'r-', lw=2)  # 画轨迹
            plt.axis('equal')
            plt.xlim([-25,25])
            plt.ylim([-30,30])
            # plt.xlim([ego_veh_state.x() - 5, ego_veh_state.x() + 100])
            # plt.ylim([ego_veh_state.y() - 5, ego_veh_state.y() + 5])
            plt.pause(0.01)
            # 将Matplotlib图形保存为图像
            if save_video:
                fig = plt.gcf()
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(np.append(np.flip(fig.get_size_inches()) * fig.dpi, 3).astype(np.int))
                if video_writer is None:
                    video_writer = cv2.VideoWriter(video_name, fourcc, frame_rate, (image.shape[1], image.shape[0]))
                video_writer.write(image)

            # 控制+定位，假设完美控制到下一个轨迹点
            ego_veh_state.transform.location.x, ego_veh_state.transform.location.y, ego_veh_state.transform.rotation.yaw \
                = traj.x[1], traj.y[1], traj.heading[1]
            ego_veh_state.kinematics.speed, ego_veh_state.kinematics.acceleration, ego_veh_state.kinematics.curvature \
                = traj.v[1], traj.a[1], traj.curvature[1]



        if save_video:
            video_writer.release()
            print("video_writer.release()")


    intersection()
