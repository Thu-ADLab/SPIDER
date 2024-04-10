from abc import abstractmethod
import warnings
import numpy as np

import spider

class BasePlanner:
    def __init__(self, config=None, *args, **kwargs):
        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

        self.local_map = None


        # 数据闭环
        self._activate_log_buffer:bool = False
        self._log_buffer:spider.data.LogBuffer = None


    @classmethod
    def default_config(cls) -> dict:
        """
        :return: a configuration dict
        """
        return {
            "output": spider.OUTPUT_TRAJECTORY,
            "steps": 20,
            "dt": 0.2,
            "ego_veh_width": 2.0,
            "ego_veh_length": 5.0,
            "max_acceleration": 6.0,
            "max_deceleration": 10.0,
            "print_info": True
        }

    @property
    def steps(self):
        return self.config.get("steps", 0)

    @property
    def dt(self):
        return self.config.get("dt", 0.0)

    @property
    def horizon(self):
        return self.steps * self.dt

    @property
    def width(self):
        return self.config.get("ego_veh_width", 5.0)

    @property
    def length(self):
        return self.config.get("ego_veh_length", 2.0)

    @property
    def log_buffer(self):
        return self._log_buffer
        # if self._activate_log_buffer:
        #     return self._log_buffer
        # else:
        #     return None

    def set_log_buffer(self, log_buffer=None):
        self._activate_log_buffer = True #not self._activate_log_buffer if enable is None else enable
        if log_buffer is not None:
            self._log_buffer = log_buffer

    def toggle_log_buffer(self, enable:bool=None):
        self._activate_log_buffer = not self._activate_log_buffer if enable is None else enable


    def configure(self, config: dict):
        warnings.warn("Method configure() is going to be deprecated. Re-instantiate a planner instead! ",
                      DeprecationWarning)
        self.__init__(config)

    @abstractmethod
    def plan(self, ego_veh_state, obstacles, routed_local_map):
        pass

    def set_local_map(self, local_map:spider.elements.RoutedLocalMap):
        self.local_map = local_map

    def __getattr__(self, key):
        if hasattr(self, "config") and key in self.config:
            return self.config[key]
        else:
            raise AttributeError("No attribute {}".format(key))

    @staticmethod
    def _update_config(default_config:dict, config:dict)->dict:
        temp = default_config.copy()
        temp.update(config)
        return temp

class DummyPlanner(BasePlanner):
    def __init__(self, config=None):
        super().__init__(config)

    def default_config(cls) -> dict:
        return cls._update_config(super().default_config(),{
            # "acceleration": ,
            "target_speed": 60/3.6,
            "max_acceleration": 6.0,
            "max_deceleration": 10.0
        })


    def plan(self, ego_veh_state:spider.elements.VehicleState, obstacles, routed_local_map=None):
        # 初始状态
        ego = ego_veh_state
        x, y = ego.x(), ego.y()
        cosyaw = np.cos(ego.yaw())
        sinyaw = np.sin(ego.yaw())
        vx, vy = ego.v() * cosyaw, ego.v() * sinyaw
        ax, ay = ego.a() * cosyaw, ego.a() * sinyaw

        # 匀加速度直线运动
        # react_time = min([self.horizon, 1.2])
        acc = (self.config["target_speed"] - ego.v()) / self.horizon
        acc = np.clip(acc, -self.config["max_deceleration"], self.config["max_acceleration"])
        ax, ay = acc * cosyaw, acc * sinyaw
        jx, jy = 0,0

        # 匀加加速度直线运动
        # react_time = min([self.horizon, 1.2])
        # avg_acc = (self.config["target_speed"] - ego.v()) / self.horizon
        # acc_end = np.clip(avg_acc - ego.a(), -self.config["max_deceleration"], self.config["max_acceleration"])
        # jerk = (acc_end - ego.a()) / self.horizon
        # jx, jy = jerk * cosyaw, jerk * sinyaw

        # 计算轨迹
        ts = np.arange(self.steps) * self.dt
        xs = x + vx * ts + 0.5 * ax * ts**2 + jx * ts**3 / 6
        ys = y + vy * ts + 0.5 * ay * ts**2 + jy * ts**3 / 6
        traj = spider.elements.Trajectory.from_trajectory_array(np.array([xs, ys]).T, dt=self.dt,
            calc_derivative=True, v0=ego.v(), heading0=ego.yaw(), a0=ego.a()
        )

        if self.config["print_info"]:
            print("\rCurrent speed: {}, Target speed: {}".format(ego.v(), self.config["target_speed"])
                   # +"\nCurrent acceleration: {}, Target acceleration: {}".format(ego.a(),acc_end)
                  )
        return traj


if __name__ == '__main__':
    a = BasePlanner()
    print(a.ego_veh_length)
    a.ego_veh_length = 10
