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


    # todo:这里不对，environment应该和reward 以及 done 解耦开！ 因为实际上reward以及done与否都是取决于人类评判的标准，不是环境客观决定的
    def is_done(self) -> bool:
        return False

    def calc_reward(self) -> float:
        return 0.0

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

    def is_done(self):
        return self.ego_veh_state.x() > 250

