

from spider.planner_zoo.BasePlanner import DummyPlanner
from spider.planner_zoo.FallbackPlanner import FallbackPlanner
from spider.utils.collision.CollisionChecker import BoxCollisionChecker
import numpy as np


class FallbackDummyPlanner(DummyPlanner):
    def __init__(self, config=None):
        super().__init__(config)
        self.fallback_planner = FallbackPlanner(config)
        self.collision_checker = BoxCollisionChecker(self.config["ego_veh_length"], self.config["ego_veh_width"])

    @classmethod
    def default_config(cls) -> dict:
        return cls._update_config(super().default_config(), FallbackPlanner.default_config())

    def plan(self, ego_veh_state, obstacles, routed_local_map=None):
        traj = super().plan(ego_veh_state, obstacles, routed_local_map)
        if (traj is None):
            traj = self.fallback_planner.plan(ego_veh_state, obstacles, routed_local_map)
        else:
            ts = np.arange(self.steps) * self.dt
            obstacles.predict(ts[ts < self.config["min_TTC"]])
            if self.collision_checker.check_trajectory(traj, obstacles):
                traj = self.fallback_planner.plan(ego_veh_state, obstacles, routed_local_map)

        return traj
