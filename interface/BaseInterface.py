from abc import abstractmethod

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
    def __init__(self):
        super(DummyInterface, self).__init__()
        self.ego_veh_state = None
        self.obstacles = None
        self.local_map = None


    def reset(self):
        # todo:应该反过来， benchmark从interface获取。
        from spider.interface.BaseBenchmark import DummyBenchmark
        self.ego_veh_state, self.obstacles, self.local_map = DummyBenchmark.get_environment_presets()
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

        vis.draw_ego_history(self.ego_veh_state, '-', lw=1, color='gray')  # 画自车历史
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
        plt.pause(0.001)


    # def is_done(self):
    #     return self.ego_veh_state.x() > 250

