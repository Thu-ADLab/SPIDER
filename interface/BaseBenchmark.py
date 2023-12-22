from abc import abstractmethod

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

    @abstractmethod
    def initial_environment(self):
        '''
        根据给定的config，初始化环境self.env
        '''
        pass

    @abstractmethod
    def test(self, spider_planner, show_video:bool=False, save_video:bool=True):
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



