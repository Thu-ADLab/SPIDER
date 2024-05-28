import warnings

import numpy as np

import spider
from spider.planner_zoo import BasePlanner
from spider.utils.collision import BoxCollisionChecker
from spider.constraints import CartConstriantChecker



class FallbackPlanner(BasePlanner):
    def __init__(self, config=None):
        super().__init__(config)
        self.constraint_checker = CartConstriantChecker(
            self.config, BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])
        )

    @classmethod
    def default_config(cls) -> dict:
        return cls._update_config(super().default_config(),{
            # "acceleration": ,
            "min_TTC": 2.0,
            "max_deceleration": 10.0,
        })


    def plan(self, ego_veh_state:spider.elements.VehicleState, obstacles:spider.elements.TrackingBoxList,
             routed_local_map=None):

        if self.config["print_info"]:
            print("WARNING: Fallback planner activated!")

        # 初始状态
        ego = ego_veh_state
        x, y = ego.x(), ego.y()
        vx, vy = ego.velocity.x, ego.velocity.y
        cosyaw = vx / ego.v()
        sinyaw = vy / ego.v()
        # vx, vy = ego.v() * cosyaw, ego.v() * sinyaw
        # ax, ay = ego.a() * cosyaw, ego.a() * sinyaw

        acc = -self.config["max_deceleration"]
        ax, ay = acc * cosyaw, acc * sinyaw

        # 计算轨迹
        ts = np.arange(self.steps) * self.dt
        t_stop = ego.v() / abs(acc)
        if t_stop > self.horizon: # can not stop in time
            xs = x + vx * ts + 0.5 * ax * ts ** 2
            ys = y + vy * ts + 0.5 * ay * ts ** 2
        else:  # has stopped at some timepoint during the horizon
            pre_idx = ts<t_stop # before stopping time
            xs, ys = np.empty_like(ts), np.empty_like(ts)
            xs[pre_idx] = x + vx * ts[pre_idx] + 0.5 * ax * ts[pre_idx] ** 2
            ys[pre_idx] = y + vy * ts[pre_idx] + 0.5 * ay * ts[pre_idx] ** 2
            xs[~pre_idx] = x + vx*t_stop*0.5
            ys[~pre_idx] = y + vy*t_stop*0.5

        traj = spider.elements.Trajectory.from_trajectory_array(np.array([xs, ys]).T, dt=self.dt,
            calc_derivative=True, v0=ego.v(), heading0=ego.yaw(), a0=ego.a()
        )

        obstacles.predict(ts[ts<self.config["min_TTC"]])
        if not self.constraint_checker.check(traj, obstacles):
            traj = None

        if self.config["print_info"]:
            if traj is None:
                print("WARNING: Fallback Planner failed to plan a collision-free trajectory within {} seconds!".format(self.config["min_TTC"]))
            else:
                print("Fallback Planner done.")

        return traj

