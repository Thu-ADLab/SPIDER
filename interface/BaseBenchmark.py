from abc import abstractmethod
from copy import deepcopy

class BaseBenchmark:
    def __init__(self, config=None):

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self.env = None # wrapped environment
        # self.metrics = {}
        # self.initial_environment()

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "max_steps": 100,
            "random_seed": 666,
            "offscreen_rendering": False,
            "save_video": True,
            "video_root": './videos/',
            "video_name": 'benchmark.mp4'
        }

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def initial_environment(self):
        '''
        根据给定的config，初始化环境self.env
        '''
        pass

    @abstractmethod
    def test(self, spider_planner):
        '''
        给定一个planner，在设置好的环境里面开一遍，返回config中指定的metrics
        '''
        pass

    @abstractmethod
    def update_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def visualize_plan(self, *args, **kwargs):
        '''
        把规划结果画在仿真器渲染的画面上
        '''
        pass




class DummyBenchmark(BaseBenchmark):
    def __init__(self, config=None):
        super(DummyBenchmark, self).__init__(config)

        from spider.interface.BaseInterface import DummyInterface

        # self.ego_veh_state = None
        # self.obstacles = None
        # self.local_map = None

        self._debug = self.config["debug_mode"]
        self._evaluation = self.config["evaluation"]
        self._max_duration = 150
        self._destination_x_range = (250, 260)
        self._destination_y_range = (-10, 10)

        self.env = DummyInterface(self.config)
        if self.config["evaluation"]:
            self.init_metric_evaluators()

    def init_metric_evaluators(self):
        from spider.interface.metrics_collection import (MetricCombiner, CompletionMetric, CollisionRateMetric,
                                                         TTCMetric, SpeedMetric, JerkMetric, StuckMetric)
        self.metric_evaluators = MetricCombiner(
            CompletionMetric(self._destination_x_range, self._destination_y_range, self._max_duration),
            CollisionRateMetric(self.config["ego_veh_length"], self.config["ego_veh_width"]),
            TTCMetric(self.config["ego_veh_length"] / 2),
            SpeedMetric(),
            JerkMetric(delta_t=0.2),  # 这里要改成和planner的dt一致
            StuckMetric(0.2),
        )

    @property
    def metrics(self):
        if self.config["evaluation"]:
            return self.metric_evaluators.get_result()
        else:
            return {}

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "random_seed": 666,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            "racetrack": "curve",  # "curve" or "straight

            "debug_mode" : False,
            "evaluation": False,
            "collision_termination": False,
            "map_frequency": 0, # 几帧更新一次map，0表示仅更新一次

            "rendering": True,
            "snapshot": True,
            "save_video": False,
            # "video_path": None,
            "video_root": './',
            "video_name": None#'benchmark.mp4'
        }


    def set_rendering(self, rendering=True):
        self.config["rendering"] = rendering

    def set_snapshot(self, snapshot=True):
        self.config["snapshot"] = snapshot


    def test(self, spider_planner, episodes=1, log=False):
        '''
        给定一个planner，在设置好的环境里面开一遍，返回config中指定的metrics
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        import spider.visualize as vis

        if self.config["evaluation"]:
            self.init_metric_evaluators()

        for episode in range(episodes):
            self.env.reset()
            if self.config["evaluation"]:
                self.metric_evaluators.reset()

            spider_planner.set_local_map(self.env.local_map)

            if self.config["rendering"]:
                vis.figure(figsize=(14, 4))

                if self.config["snapshot"]:
                    video_name = type(spider_planner).__name__ + '.avi' if self.config["video_name"] is None else self.config["video_name"]
                    snapshot = vis.SnapShot(True, 15, record_video=self.config["save_video"],
                                            video_path=self.config["video_root"]+video_name)
            ################## main loop ########################
            try:
                for i in range(self._max_duration):
                # while True:
                # 评估
                    if self.env.ego_veh_state.x() > self._destination_x_range[0]:
                        break


                    # 地图信息更新
                    if self.config["map_frequency"] == 0:
                        local_map = None
                    else:
                        if i % self.config["map_frequency"] == 0:
                            local_map = deepcopy(self.env.local_map)
                        else:
                            local_map = None

                    # 感知信息更新，这里假设完美感知+其他车全部静止

                    # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
                    # obs = deepcopy(self.env.wrap_observation())
                    # obs[-1] = local_map
                    traj = spider_planner.plan(deepcopy(self.env.ego_veh_state), deepcopy(self.env.obstacles), local_map)  # , self.local_map

                    if traj is None:
                        raise RuntimeError("DummyBenchmark receives no feasible trajectory!")

                    # 可视化
                    if self.config["rendering"]:
                        plt.cla()
                        self.env.visualize(traj)
                        plt.pause(0.001)

                        if self.config["snapshot"]:
                            snapshot.snap(plt.gca())

                    # 控制+定位，假设完美控制到下一个轨迹点
                    self.env.conduct_trajectory(traj)

                    # 评估
                    if self.config["evaluation"]:
                        self.metric_evaluators.evaluate(self.env.ego_veh_state, self.env.obstacles, local_map)

            except Exception as e:
                if self._debug:
                    raise e
                print(e)

            finally:


                if self.config["rendering"]:
                    plt.close()
                    if self.config["snapshot"]:
                        snapshot.print(3, 2, figsize=(15, 6))
                        plt.show()

        if self.config["evaluation"]:
            print(self.metrics)

    @staticmethod
    def get_environment_presets(ego_length=5.0, ego_width=2.0, racetrack="curve"):
        from spider.interface.BaseInterface import DummyInterface
        return DummyInterface.get_environment_presets(ego_length, ego_width, racetrack)

    def update_metrics(self, *args, **kwargs):
        pass
    #
    #
    # def visualize_plan(self, *args, **kwargs):
    #     '''
    #     把规划结果画在仿真器渲染的画面上
    #     '''
    #     pass

if __name__ == '__main__':
    from spider.planner_zoo import LatticePlanner, BezierPlanner, PiecewiseLatticePlanner
    planner = LatticePlanner({
        "steps": 15,
        "dt": 0.2,
        "end_l_candidates": (-3.5, 0, 3.5),
    })

    # planner = PiecewiseLatticePlanner({
    #     "steps": 15,
    #     "dt": 0.2,
    # })

    benchmark = DummyBenchmark()
    benchmark.test(planner)

