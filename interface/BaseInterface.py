from abc import abstractmethod
import copy

class BaseInterface:
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        '''
        randomly (or not) reset the environment
        '''
        pass

    @abstractmethod
    def wrap_observation(self):
        pass

    @abstractmethod
    def conduct_trajectory(self, trajectory):
        pass

    @abstractmethod
    def convert_to_action(self, planner_output):
        pass


    # environment应该和reward 以及 done 解耦开！ 因为实际上reward以及done与否都是取决于人类评判的标准，不是环境客观决定的
    # def is_done(self) -> bool:
    #     return False
    #
    # def calc_reward(self) -> float:
    #     return 0.0

class DummyInterface(BaseInterface):
    def __init__(self,config=None):
        super(DummyInterface, self).__init__()
        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self.ego_veh_state = None
        self.obstacles = None
        self.local_map = None

        assert self.config["racetrack"] in ["curve", "straight"], "racetrack must be one of ['curve', 'straight']"
        self.init_observation = self.get_environment_presets(
            self.config["ego_veh_length"], self.config["ego_veh_width"], self.config["racetrack"])
        self.reset()

    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            # "random_seed": 666,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,

            "racetrack": "curve",  # "curve" or "straight
        }


    def reset(self, random=False):
        self.ego_veh_state, self.obstacles, self.local_map = copy.deepcopy(self.init_observation)
        ##
        if random:
            import random as rnd
            dx, dy = rnd.random() * 6 - 3, rnd.random() * 6 - 3
            dv = rnd.random() * 0.5
            self.ego_veh_state.transform.location.x += dx
            self.ego_veh_state.transform.location.y += dy
            self.ego_veh_state.velocity.y += dv
        ####
        return self.wrap_observation()


    def wrap_observation(self):
        return self.ego_veh_state, self.obstacles, self.local_map


    def convert_to_action(self, planner_output):
        return planner_output

    def conduct_trajectory(self, trajectory):
        traj = trajectory
        # 控制+定位，假设完美控制到下一个轨迹点
        self.ego_veh_state.transform.location.x, self.ego_veh_state.transform.location.y, self.ego_veh_state.transform.rotation.yaw \
            = traj.x[1], traj.y[1], traj.heading[1]
        self.ego_veh_state.kinematics.speed, self.ego_veh_state.kinematics.acceleration, self.ego_veh_state.kinematics.curvature \
            = traj.v[1], traj.a[1], traj.curvature[1]

        for tb in self.obstacles:
            tb.set_obb([tb.x + tb.vx * traj.dt, tb.y + tb.vy * traj.dt, tb.length, tb.width, tb.box_heading])


    def visualize(self,traj):
        '''
        qzl: 要修改，统一格式
        '''
        import spider.visualize as vis
        import matplotlib.pyplot as plt
        import numpy as np

        for lane in self.local_map.lanes:
            plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图
        # vis.draw_ego_vehicle(ego_veh_state, color='green', fill=True, alpha=0.2, linestyle='-', linewidth=1.5) # 画自车

        for tb in self.obstacles:
            vis.draw_boundingbox(tb, color='black', fill=True, alpha=0.1, linestyle='-', linewidth=1.5)  # 画他车
            # 画他车预测轨迹
            tb_pred_traj = np.column_stack((tb.x + np.asarray(traj.t) * tb.vx, tb.y + np.asarray(traj.t) * tb.vy))
            vis.draw_polyline(tb_pred_traj, show_buffer=True, buffer_dist=tb.width * 0.5, buffer_alpha=0.1,
                              color='C3')

        # vis.draw_ego_history(self.ego_veh_state, '-', lw=1, color='gray')  # 画自车历史
        vis.draw_trajectory(traj, '.-', show_footprint=True, color='C2')  # 画轨迹
        if "control_points" in traj.debug_info:  # bezier planner
            pts = traj.debug_info["control_points"]
            plt.plot(pts[:, 0], pts[:, 1], 'or')
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
        # plt.pause(0.001)

    @staticmethod
    def get_environment_presets(ego_length=5.0, ego_width=2.0, racetrack="curve"):
        ############ environment presets ###################
        import numpy as np
        from spider.elements.map import RoutedLocalMap, Lane
        from spider.elements.box import TrackingBoxList, TrackingBox
        from spider.elements.vehicle import VehicleState

        ## localization
        init_ego_state = VehicleState.from_kine_states(5., 1., 0., vx=0.5, vy=0.,
                                                       length=ego_length, width=ego_width)
        # init_ego_state = VehicleState.from_kine_states(5., 1., 0., vx=0.5, vy=0.,
        #                                                      length=ego_length, width=ego_width)

        ### map
        init_local_map = RoutedLocalMap()
        for idx, yy in enumerate([-3.5, 0, 3.5]):
            xs = np.arange(0, 300.1, 1.0)
            if racetrack == "curve":
                cline = np.column_stack((xs, 3 * np.sin(np.pi * xs / 100) + yy))  # sin形状车道
            else:
                cline = np.column_stack((xs, np.ones_like(xs) * yy))  # 直线车道

            lane = Lane(idx, cline, width=3.5, speed_limit=60 / 3.6)
            init_local_map.lanes.append(lane)

        ## perception
        init_obstacles = TrackingBoxList([
            TrackingBox(obb=(50, 0, 5, 2, np.arctan2(0.2, 5)), vx=5, vy=0.2),
            TrackingBox(obb=(100, 0, 5, 2, np.arctan2(-0.2, 5)), vx=5, vy=-0.25),
            TrackingBox(obb=(200, -10, 1, 1, np.pi / 2), vx=0, vy=1.0),  # 横穿马路
            # TrackingBox(obb=(220, -2, 1, 1, -np.pi / 2), vx=0.0, vy=0.0)  # 路障
        ])
        return init_ego_state, init_obstacles, init_local_map

