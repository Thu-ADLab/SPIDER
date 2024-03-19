

class BasePlanner:
    def __init__(self, config=None, *args, **kwargs):
        self.config = {} if config is None else config

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
        self.__init__(config)

