import math
from typing import List
import time
import numpy as np
import warnings

import spider
from spider.planner_zoo.BasePlanner import BasePlanner

from spider.elements.map import RoutedLocalMap
from spider.elements.trajectory import FrenetTrajectory
from spider.elements.vehicle import VehicleState
from spider.elements.box import TrackingBoxList, TrackingBox

from spider.sampler.PolynomialSampler import QuarticPolyminalSampler, QuinticPolyminalSampler,  PiecewiseQuinticPolyminalSampler
from spider.sampler.Combiner import LatLonCombiner
from spider.evaluator import FrenetCostEvaluator

from spider.utils.transform.frenet import FrenetCoordinateTransformer
from spider.utils.collision import BoxCollisionChecker

from spider.constraints import CartConstriantChecker


class PiecewiseLatticePlanner(BasePlanner):
    def __init__(self, config=None):
        super(PiecewiseLatticePlanner, self).__init__()

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)
            # self.configure(config)

        self.local_map = RoutedLocalMap()
        self.coordinate_transformer = FrenetCoordinateTransformer() # 要维护几个坐标系呢？
        # self.predictor = None
        self.longitudinal_sampler = QuarticPolyminalSampler(self.config["end_T_candidates"],
                                                            self.config["end_v_candidates"])
        self.lateral_sampler = PiecewiseQuinticPolyminalSampler(
            self.config["delta_s"], self.config["max_seg_num"], self.config["l_candidates"]
        )
        
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
            "safe_distance": (1.0, 0.2),
            "delta_s": 12,
            "max_seg_num": 3,
            "l_candidates": (-3.5,0,3.5), # s,d采样生成横向轨迹 (-3.5, 0, 3.5), #

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

    def get_candidate_traj_with_cost(self):
        return self._candidate_trajectories, self._candidate_trajectories_cost


    def constraint_check(self, sorted_candidate_trajectories:List[FrenetTrajectory], sorted_cost, obstacles:TrackingBoxList):
        for traj, cost in zip(sorted_candidate_trajectories, sorted_cost):
            traj = self.coordinate_transformer.frenet2cart4traj(traj, order=2) # 在笛卡尔坐标系下做碰撞检测
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
        # todo: qzl:这一步耗时非常严重，有2个建议：
        #  1. 评估暂时用不到笛卡尔坐标，碰撞检测目前在笛卡尔坐标下，所以可以在碰撞检测的时候再转换，不用提前把所有的都做坐标转换
        #  2. 把碰撞检测直接整个放在Frenet坐标下，即先把障碍物及其预测轨迹变换到Frenet坐标，这样子所有的环节都在Frenet坐标进行，最后输出前再转换到笛卡尔
        #  p.s.第一点的补充：坐标转换可以暂时先存储成函数或生成器，暂时先不执行，在碰撞检测的时候才执行
        # candidate_trajectories = [self.coordinate_transformer.frenet2cart4traj(t, order=2) for t in candidate_trajectories]

        # 评估+筛选
        sorted_candidates, sorted_cost = self.trajectory_evaluator.evaluate_candidates(candidate_trajectories)
        optimal_trajectory, min_cost = self.constraint_check(sorted_candidates, sorted_cost, predicted_obstacles)

        self._candidate_trajectories, self._candidate_trajectories_cost = sorted_candidates, sorted_cost

        if not (optimal_trajectory is None):
            print("Optimal trajectory found! s_dot_end=%.2f,l_end=%.2f" %
                  (optimal_trajectory.s_dot[-1], optimal_trajectory.l[-1]))
        else:
            warnings.warn("WARNING: NO feasible trajectory!")

        t2 = time.time()
        print("Planning Succeed! Time: %.2f seconds, FPS: %.2f" % (t2 - t1, 1 / (t2 - t1)))

        return optimal_trajectory

