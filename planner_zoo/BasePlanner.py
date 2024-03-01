

class BasePlanner:
    def __init__(self, config=None, *args, **kwargs):
        self.config = {}

    @property
    def steps(self):
        return self.config["steps"]

    @property
    def dt(self):
        return self.config["dt"]

    def configure(self, config: dict):
        self.__init__(config)

