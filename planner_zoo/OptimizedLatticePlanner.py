import warnings

import numpy as np
import copy

from spider.planner_zoo.LatticePlanner import LatticePlanner
from spider.optimize.TrajectoryOptimizer import FrenetTrajectoryOptimizer

from spider.elements import RoutedLocalMap, TrackingBoxList, VehicleState
from spider.control.vehicle_model import Bicycle

from spider.utils.collision import BoxCollisionChecker
from spider.constraints import CartConstriantChecker

# todo：把优化器的打印输出增加一个可关闭选项，把lattice planner内部的计时和输出打印关闭，增加一个全部的计时打印输出功能
class OptimizedLatticePlanner(LatticePlanner):
    def __init__(self, config=None):
        if "steps" in config or "dt" in config:
            warnings.warn("ATTENTION! Change of steps and dt needs the adjustment of optimizer parameters!")

        super(OptimizedLatticePlanner, self).__init__(config)

        self.constraint_checker = CartConstriantChecker( # 由于优化问题暂时视作质点优化问题，为了保证采样的初始解是可行解，需要使用质点约束检查器
            self.config, BoxCollisionChecker(0.1, 0.1) # 留着0.2是为了安全冗余
        )
        self.optimizer = FrenetTrajectoryOptimizer(self.steps, self.dt)
        self.initial_traj = None

    @property
    def corridor(self):
        return self.optimizer.corridor

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = super().default_config()
        config.update({
            "steps": 20,
            "dt": 0.2,
            "end_l_candidates": (-3.5, 0, 3.5),
            "wheelbase": 3.0
        })
        return config

    # def configure(self, config: dict):
    #
    #     super(OptimizedLatticePlanner, self).configure(config)



    def plan(self, ego_veh_state:VehicleState, obstacles:TrackingBoxList, local_map:RoutedLocalMap=None):
        # 将obstacles膨胀，以使得优化问题变为一个质点的轨迹规划问题
        obstacles = obstacles.dilate(self.length, self.width)

        traj = super().plan(ego_veh_state, obstacles, local_map)
        self.initial_traj = copy.deepcopy(traj)

        if traj is None:
            if self.config["print_info"]:
                print("No feasible sampling initial trajectory! ")
            return None


        # 补充l_dot和l_2dot信息，因为本身以l_prime和l_2prime存储
        traj.l_dot = np.asarray(traj.l_prime) * np.asarray(traj.s_dot)
        traj.l_2dot = np.asarray(traj.l_2prime) * np.asarray(traj.s_dot) ** 2 \
                          + np.asarray(traj.l_prime) * np.asarray(traj.s_2dot)


        # 将obstacles转化到frenet坐标下
        obstacles_with_fstate = self.coordinate_transformer.cart2frenet4boxes(obstacles, convert_prediction=True, order=0)

        # 优化s,l序列
        opt_traj = self.optimizer.optimize_traj(traj, obstacles_with_fstate)  # 仅有s, l信息
        # 将sl序列转化为xy序列
        opt_traj = self.coordinate_transformer.frenet2cart4traj(opt_traj, order=0)  # s,l -> x,y
        # xy序列微分，提取高阶运动学信息
        opt_traj.derivative(
            Bicycle(ego_veh_state.x(), ego_veh_state.y(), ego_veh_state.v(), ego_veh_state.a(),
                    ego_veh_state.yaw(), dt=self.dt, wheelbase=self.config["wheelbase"]),
            opt_traj.x, opt_traj.y)  # 补充高阶导数信息

        # 附加debug信息corridor，供调试和可视化
        opt_traj.debug_info["corridor"] = self.corridor
        opt_traj.debug_info["initial_trajectory"] = self.initial_traj
        return opt_traj


if __name__ == '__main__':
    from spider.interface.BaseBenchmark import DummyBenchmark

    bm = DummyBenchmark()
    # bm._init_obstacles = TrackingBoxList([
    #     TrackingBox(obb=(50, 0, 5, 2, np.arctan2(0.2, 5))),
    #     TrackingBox(obb=(100, 0, 5, 2, np.arctan2(-0.2, 5))),
    #     TrackingBox(obb=(200, -10, 1, 1, np.pi / 2))  # 横穿马路
    # ])

    # planner = LatticePlanner({
    #     "steps": 20,
    #     "dt": 0.2,
    # })
    planner = OptimizedLatticePlanner({
        "steps": 20,
        "dt": 0.2,
    })
    bm.test(planner)

