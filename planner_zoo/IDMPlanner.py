import warnings

import numpy as np
import math

import spider
from spider.planner_zoo import BasePlanner
# from spider.utils.collision import BoxCollisionChecker
# from spider.constraints import CartConstriantChecker

from spider.control.IDMController import IDMController
from spider.utils.transform import FrenetTransformer

class IDMPlanner(BasePlanner):
    def __init__(self, config=None):
        super().__init__(config)
        self.idm_controller = IDMController()

        self.frenet_tf = FrenetTransformer()

    @classmethod
    def default_config(cls) -> dict:
        return cls._update_config(super().default_config(),{
            # "min_TTC": 2.0,
            # "max_deceleration": 10.0,
        })


    def plan(self, ego_veh_state:spider.elements.VehicleState, obstacles:spider.elements.TrackingBoxList,
             local_map=None):

        if not (local_map is None):
            self.set_local_map(local_map)

        ego_lane_idx , (ego_s, _) = self.local_map.match_lane(ego_veh_state.x(), ego_veh_state.y(), return_frenet=True)
        target_lane = self.local_map.lanes[ego_lane_idx]
        ref_line = target_lane.centerline

        self.frenet_tf.set_reference_line(target_lane.centerline, target_lane.centerline_csp)
        roi_range = target_lane.width / 2.0 + 0.5
        boxes_with_frenet = self.frenet_tf.cart2frenet4boxes(obstacles, order=1)

        front_veh_speed, front_veh_dist = 60 / 3.6, 1000
        for tb in boxes_with_frenet:
            if -roi_range < tb.frenet_state.l < roi_range and tb.frenet_state.s > ego_s:
                dist = tb.frenet_state.s - ego_s
                if dist < front_veh_dist:
                    front_veh_dist = dist
                    front_veh_speed = tb.frenet_state.s_dot


        # 初始状态
        ego = ego_veh_state
        x, y = ego.x(), ego.y()
        vx, vy = ego.velocity.x, ego.velocity.y
        cosyaw = vx / ego.v()
        sinyaw = vy / ego.v()

        # 控制量计算
        current_pose = (x, y, ego.yaw())
        current_speed = ego.v()
        desired_speed = target_lane.speed_limit
        acc, steer = self.idm_controller.get_control(ref_line, front_veh_speed, front_veh_dist, desired_speed,
                                              current_pose, current_speed)
        # ax, ay = acc * cosyaw, acc * sinyaw

        # 计算轨迹
        accs = acc*np.ones(self.steps-1)
        steers = steer*np.ones(self.steps-1)
        veh_model = spider.control.Bicycle(x, y, ego.v(), ego.a(), ego.yaw(), dt=self.dt)
        traj = spider.elements.Trajectory(self.steps, self.dt)
        traj.step(veh_model, accs, steers)

        if self.config["print_info"]:
            print("IDMPlanner: front_veh: dist, speed = ", front_veh_dist, front_veh_speed)
            print("IDMPlanner: acc, steer = ", acc, steer)

        return traj

if __name__ == '__main__':
    from spider.interface import DummyBenchmark
    planner = IDMPlanner()
    benchmark = DummyBenchmark({
        "debug_mode": True,
    })

    benchmark.test(planner)

