from abc import abstractmethod
import warnings

import spider

class BasePlanner:
    def __init__(self, config=None, *args, **kwargs):
        self.config = self.default_config()
        if not (config is None):
            self.config.update(config)

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
            "ego_veh_length": 5.0
        }

    @property
    def steps(self):
        return self.config.get("steps", 0)

    @property
    def dt(self):
        return self.config.get("dt", 0.0)

    @property
    def width(self):
        return self.config.get("ego_veh_width", 0.0)

    @property
    def length(self):
        return self.config.get("ego_veh_length", 0.0)

    def configure(self, config: dict):
        warnings.warn("Method configure() is going to be deprecated. Re-instantiate a planner instead! ",
                      DeprecationWarning)
        self.__init__(config)

    @abstractmethod
    def plan(self, ego_veh_state, obstacles, routed_local_map):
        pass

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError("No attribute {}".format(key))

    def __setattr__(self, key, value):
        if key in self.config:
            self.config[key] = value
        else:
            raise AttributeError("No attribute {}".format(key))



if __name__ == '__main__':
    a = BasePlanner()
    print(a.ego_veh_length)
    a.ego_veh_length = 10
