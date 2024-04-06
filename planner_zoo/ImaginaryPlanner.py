import time

import warnings
import spider
from spider.planner_zoo.BasePlanner import BasePlanner

from spider.elements.map import RoutedLocalMap
from spider.elements.trajectory import FrenetTrajectory
from spider.elements.vehicle import VehicleState
from spider.elements.box import TrackingBoxList, TrackingBox

from spider.utils.ImaginaryEngine import ImaginaryEngine
from spider.planner_zoo.LatticePlanner import LatticePlanner


class ImaginaryPlanner(BasePlanner):
    def __init__(self, config=None):
        super(ImaginaryPlanner, self).__init__(config)

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self._atom_planner = LatticePlanner(self.config["planner_config"])
        self._predictor = None
        self._tracker = None
        self.track_steps = self.config["track_steps"]

        self.imaginary_engine = ImaginaryEngine(
            self.steps, self.dt, self._atom_planner,
            predictor=self._predictor, tracker=self._tracker, track_steps=self.track_steps
        )

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        config = {
            "steps": 30,
            "dt": 0.2,
            "track_steps": 5,
            "print_info": True,

            "planner_config": LatticePlanner.default_config()
        }

        config["planner_config"].update({
            "steps": 10,
            "dt": 0.2,
            "end_s_candidates": (20, 40, 60),
            "end_l_candidates": (-3.5, 0, 3.5),
            "print_info": False
        })
        return config

    # todo: 以后atom_planner, predictor和tracker用 @property和@xxx.setter来设置

    def set_atom_planner(self, atom_planner):
        self._atom_planner = atom_planner
        self.imaginary_engine.atom_planner = self._atom_planner

    def set_predictor(self, predictor):
        self._predictor = predictor
        self.imaginary_engine.predictor = self._predictor

    def set_tracker(self,tracker):
        self._tracker = tracker
        self.imaginary_engine.tracker = self._tracker

    def set_local_map(self, local_map:RoutedLocalMap):
        self.local_map = local_map


    def plan(self, ego_veh_state:VehicleState, obstacles:TrackingBoxList, local_map:RoutedLocalMap=None) -> FrenetTrajectory:
        """
        输入定位、物体、（地图optional,更新频率比较慢。建议在外面单独写set地图的逻辑）
        输出轨迹（FrenetTrajectory）
        """
        t1 = time.time()
        # 储存地图。每条车道代表了一个frenet坐标系，储存地图即储存了lanes，即储存了可供选择的几个frenet坐标系
        if not (local_map is None):
            self.set_local_map(local_map)

        traj, truncated = self.imaginary_engine.imagine(
            ego_veh_state, obstacles, self.local_map)

        if truncated:
            if self.config["print_info"]:
                warnings.warn("trajectory is truncated! No solution during imagination...")

        t2 = time.time()
        if self.config["print_info"]:
            print("Planning Succeed! Time: %.2f seconds, FPS: %.2f" % (t2 - t1, 1 / (t2 - t1)))

        return traj




