import math
from typing import List
import time
import numpy as np

import spider
from spider.planner_zoo.BasePlanner import BasePlanner

from spider.elements.map import RoutedLocalMap
from spider.elements.trajectory import Trajectory
from spider.elements.vehicle import VehicleState
from spider.elements.box import TrackingBoxList, TrackingBox

from spider.sampler import QuarticPolyminalSampler, BezierCurveSampler, PVDCombiner
from spider.evaluator import CartCostEvaluator

from spider.utils.transform.frenet import FrenetCoordinateTransformer
from spider.utils.collision import BoxCollisionChecker

from spider.constraints import CartConstriantChecker


class BezierPlanner(BasePlanner):
    def __init__(self, config=None):
        super(BasePlanner, self).__init__()

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self.local_map = RoutedLocalMap()
        self.coordinate_transformer = FrenetCoordinateTransformer() # 要维护几个坐标系呢？
        # self.predictor = None
        self.path_sampler = BezierCurveSampler(self.config["end_s_candidates"],self.config["end_l_candidates"])
        self.displacement_sampler = QuarticPolyminalSampler(self.config["end_T_candidates"],
                                                            self.config["end_v_candidates"])

        self.trajectory_combiner = PVDCombiner(self.config["steps"], self.config["dt"]) # 默认路径-速度解耦的重新耦合
        self.trajectory_evaluator = CartCostEvaluator()

        # self.collision_checker = BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])
        self.constraint_checker = CartConstriantChecker(
            self.config, BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])
        )


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 15,
            "dt": 0.2,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            "max_speed": 60/3.6,
            "min_speed": 0,
            "max_acceleration": 10,
            "max_deceleration": 10,
            # "max_centripetal_acceleration" : 100,
            "max_curvature": 100,
            "safe_distance": (1.0, 0.2),
            "end_s_candidates": (20,30),
            "end_l_candidates": (-3.5, 0, 3.5),
            "end_v_candidates": tuple(i * 60 / 3.6 / 3 for i in range(4)),  # 改这一项的时候，要连着限速一起改了
            "end_T_candidates": (2, 4, 8),  # s_dot, T采样生成纵向轨迹

            "constraint_flags": {
                spider.CONSTRIANT_SPEED_UB,
                spider.CONSTRIANT_SPEED_LB,
                spider.CONSTRIANT_ACCELERATION,
                spider.CONSTRIANT_DECELERATION,
                spider.CONSTRIANT_CURVATURE
            },
        }


    def constraint_check(self, sorted_candidate_trajectories:List[Trajectory], sorted_cost, obstacles:TrackingBoxList):
        for traj, cost in zip(sorted_candidate_trajectories, sorted_cost):
            if self.constraint_checker.check(traj, obstacles):
                return traj, cost

        return None, 0

    def set_local_map(self, local_map:RoutedLocalMap):
        self.local_map = local_map

    def build_frenet_lane(self, target_lane_idx):
        if target_lane_idx < 0 or target_lane_idx >= len(self.local_map.lanes):
            raise ValueError("Invalid target lane record_index")
        target_lane = self.local_map.lanes[target_lane_idx]
        self.coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)



    def plan(self, ego_veh_state:VehicleState, obstacles:TrackingBoxList, local_map:RoutedLocalMap=None) -> Trajectory:
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
        target_lane_idx = 1#ego_lane_idx  # 目前车道决策还没写上，先默认自车车道，按道理是一个以自车车道输入的函数
        self.build_frenet_lane(target_lane_idx)

        # 预测
        predicted_obstacles = obstacles.predict(self.config["dt"] * np.arange(self.config["steps"]))

        # 轨迹采样
        displacement_samples = self.displacement_sampler.sample((0.0, ego_veh_state.v(), ego_veh_state.a()))
        path_samples = self.path_sampler.sample(ego_veh_state.x(), ego_veh_state.y(), ego_veh_state.yaw(),
                                                self.coordinate_transformer)
        candidate_trajectories = self.trajectory_combiner.combine(path_samples, displacement_samples)

        # 评估+筛选
        sorted_candidates, sorted_cost = self.trajectory_evaluator.evaluate_candidates(candidate_trajectories)
        optimal_trajectory, min_cost = self.constraint_check(sorted_candidates, sorted_cost, predicted_obstacles)
        if not (optimal_trajectory is None):
            print("Optimal trajectory found!")
        else:
            print("WARNING: NO feasible trajectory!")

        t2 = time.time()
        print("Planning Succeed! Time: %.2f seconds, FPS: %.2f" % (t2 - t1, 1 / (t2 - t1)))

        return optimal_trajectory


if __name__ == '__main__':
    from spider.interface.BaseBenchmark import DummyBenchmark
    planner = BezierPlanner({
        "steps": 15,
        "dt": 0.2,
        "end_s_candidates": (20,30),
        "end_l_candidates": (-3.5, 0, 3.5),
        "end_v_candidates": tuple(i * 60 / 3.6 / 3 for i in range(4)),  # 改这一项的时候，要连着限速一起改了
        "end_T_candidates": (2, 4, 8),  # s_dot, T采样生成纵向轨迹
    })

    benchmark = DummyBenchmark()
    benchmark.test(planner)
