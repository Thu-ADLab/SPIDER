from abc import abstractmethod
from copy import deepcopy

class BaseBenchmark:
    def __init__(self, config=None):

        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self.env = None
        self.metrics = {}
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


    def reset(self):
        return self.initial_environment()

    @abstractmethod
    def initial_environment(self):
        '''
        根据给定的config，初始化环境self.env
        '''
        pass

    @abstractmethod
    def test(self, spider_planner, log=False):
        '''
        给定一个planner，在设置好的环境里面开一遍，返回config中指定的metrics
        '''
        self.initial_environment()

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
        self.ego_veh_state = None
        self.obstacles = None
        self.local_map = None

        ############ environment presets ###################
        import numpy as np
        from spider.elements.map import RoutedLocalMap, Lane
        from spider.elements.Box import TrackingBoxList, TrackingBox
        from spider.elements.vehicle import VehicleState, Transform, Location, Rotation, Vector3D

        ## localization
        self._init_ego_state = VehicleState.from_kine_states(5.,1.,0.,vx=0.5,vy=0.,length=self.config["ego_veh_length"],
                                                             width=self.config["ego_veh_width"])

        ### map
        self._init_local_map = RoutedLocalMap()
        for idx, yy in enumerate([-3.5, 0, 3.5]):
            xs = np.arange(0, 300.1, 1.0)
            # cline = np.column_stack((xs, np.ones_like(xs) * yy)) # 直线车道
            cline = np.column_stack((xs, 3 * np.sin(np.pi*xs/100) + yy)) # sin形状车道
            lane = Lane(idx, cline, width=3.5, speed_limit=60 / 3.6)
            self._init_local_map.lanes.append(lane)

        ## perception
        self._init_obstacles = TrackingBoxList([
            TrackingBox(obb=(50, 0, 5, 2, np.arctan2(0.2, 5)), vx=5, vy=0.2),
            TrackingBox(obb=(100, 0, 5, 2, np.arctan2(-0.2, 5)), vx=5, vy=-0.25),
            TrackingBox(obb=(200, -10, 1, 1, np.pi / 2), vx=0, vy=1.0),  # 横穿马路
            # TrackingBox(obb=(202, 10, 1, 1, -np.pi / 2), vx=0, vy=-1.0)  # 横穿马路
        ])


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "random_seed": 666,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            # "offscreen_rendering": False,
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


    def initial_environment(self):
        '''
        根据给定的config，初始化环境self.env
        '''
        #################### 输入信息的初始化 ####################
        # 定位信息
        self.ego_veh_state = deepcopy(self._init_ego_state)
        # 地图信息
        self.local_map = deepcopy(self._init_local_map)
        # 感知信息
        self.obstacles = deepcopy(self._init_obstacles)

    def test(self, spider_planner, log=False):
        '''
        给定一个planner，在设置好的环境里面开一遍，返回config中指定的metrics
        '''
        import matplotlib.pyplot as plt
        import numpy as np
        import spider.visualize as vis

        self.initial_environment()
        spider_planner.set_local_map(self.local_map)

        if self.config["rendering"]:
            vis.figure(figsize=(14, 4))

            if self.config["snapshot"]:
                video_name = type(spider_planner).__name__ + '.avi' if self.config["video_name"] is None else self.config["video_name"]
                snapshot = vis.SnapShot(True, 15, record_video=self.config["save_video"],
                                        video_path=self.config["video_root"]+video_name)
        ################## main loop ########################
        try:
            while True:
                if self.ego_veh_state.x() > 250:
                    break

                if self.config["rendering"]:
                    plt.cla()

                # 地图信息更新

                # 感知信息更新，这里假设完美感知+其他车全部静止

                # 定位信息更新,本应该放在前面从gps拿，这里直接假设完美控制，在后面从控制拿了
                # ego_veh_state = ...

                traj = spider_planner.plan(deepcopy(self.ego_veh_state), deepcopy(self.obstacles))  # , local_map)

                if traj is None:
                    raise RuntimeError("DummyBenchmark receives no feasible trajectory!")

                # 可视化
                if self.config["rendering"]:

                    for lane in self.local_map.lanes:
                        plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图
                    # vis.draw_ego_vehicle(ego_veh_state, color='green', fill=True, alpha=0.2, linestyle='-', linewidth=1.5) # 画自车

                    for tb in self.obstacles:
                        vis.draw_boundingbox(tb, color='black', fill=True, alpha=0.1, linestyle='-', linewidth=1.5)  # 画他车
                        # 画他车预测轨迹
                        tb_pred_traj = np.column_stack((tb.x + np.asarray(traj.t) * tb.vx, tb.y + np.asarray(traj.t) * tb.vy))
                        vis.draw_polyline(tb_pred_traj, show_buffer=True, buffer_dist=tb.width * 0.5, buffer_alpha=0.1,
                                          color='C3')

                    vis.draw_ego_history(self.ego_veh_state, '-', lw=1, color='gray')  # 画自车历史
                    vis.draw_trajectory(traj, '.-', show_footprint=True, color='C2')  # 画轨迹
                    if "control_points" in traj.debug_info: # bezier planner
                        pts = traj.debug_info["control_points"]
                        plt.plot(pts[:,0], pts[:,1], 'or')
                    # if "corridor" in traj.debug_info: # optimizer planner
                    #     vis.draw_corridor(traj.debug_info["corridor"], color='green', linewidth=0.5)
                    if "initial_trajectory" in traj.debug_info:
                        vis.draw_trajectory(traj.debug_info["initial_trajectory"], '--', color="black", show_footprint=False)



                    vis.draw_ego_vehicle(self.ego_veh_state, color='C0', fill=True, alpha=0.3, linestyle='-', linewidth=1.5)  # 画自车
                    # plt.axis('equal')
                    # plt.tight_layout()
                    vis.ego_centric_view(self.ego_veh_state.x(), self.ego_veh_state.y(), [-20, 80], [-5, 5])
                    # plt.xlim([ego_veh_state.x() - 20, ego_veh_state.x() + 80])
                    # plt.ylim([ego_veh_state.y() - 5, ego_veh_state.y() + 5])
                    plt.pause(0.01)

                    if self.config["snapshot"]:
                        snapshot.snap(plt.gca())

                # 控制+定位，假设完美控制到下一个轨迹点
                self.ego_veh_state.transform.location.x, self.ego_veh_state.transform.location.y, self.ego_veh_state.transform.rotation.yaw \
                    = traj.x[1], traj.y[1], traj.heading[1]
                self.ego_veh_state.kinematics.speed, self.ego_veh_state.kinematics.acceleration, self.ego_veh_state.kinematics.curvature \
                    = traj.v[1], traj.a[1], traj.curvature[1]

                for tb in self.obstacles:
                    tb.set_obb([tb.x + tb.vx * traj.dt, tb.y + tb.vy * traj.dt, tb.length, tb.width, tb.box_heading])
        except Exception as e:
            print(e)

        finally:
            if self.config["rendering"]:
                plt.close()
                if self.config["snapshot"]:
                    snapshot.print(3, 2, figsize=(15, 6))
                    plt.show()


    @classmethod
    def get_environment_presets(cls):
        benchmark = cls()
        benchmark.initial_environment()
        return benchmark.ego_veh_state, benchmark.obstacles, benchmark.local_map

    # def update_metrics(self, *args, **kwargs):
    #     pass
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

