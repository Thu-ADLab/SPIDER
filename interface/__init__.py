from spider.interface.BaseInterface import BaseInterface, DummyInterface
from spider.interface.BaseBenchmark import BaseBenchmark, DummyBenchmark

import spider
try: # 这个写的不是特别好， 暂时先这样子
    from spider.interface.highway_env import HighwayEnvInterface, HighwayEnvBenchmark, HighwayEnvBenchmarkGUI
except (ModuleNotFoundError, ImportError) as e:
    HighwayEnvInterface = HighwayEnvBenchmark = HighwayEnvBenchmarkGUI = spider._virtual_import("highway_env", e)

try:
    from spider.interface.carla import CarlaInterface
except (ModuleNotFoundError, ImportError) as e:
    CarlaInterface = spider._virtual_import("carla", e)



# __all__ = [
#     "BaseInterface",
#     "DummyInterface",
#     "BaseBenchmark",
#     "DummyBenchmark",
#     "HighwayEnvInterface",
#     # "HighwayEnvBenchmark",
#     # "HighwayEnvBenchmarkGUI",
#     # "CarlaInterface"
# ]

# def __getattr__(name):
#     if name == "HighwayEnvInterface":
#         from spider.interface.highway_env import HighwayEnvInterface
#         return HighwayEnvInterface
#     elif name == "HighwayEnvBenchmark":
#         return HighwayEnvBenchmark
#     elif name == "HighwayEnvBenchmarkGUI":
#         return HighwayEnvBenchmarkGUI
#     elif name == "CarlaInterface":
#         return CarlaInterface
#     else:
#         raise AttributeError("Module 'spider.interfaces' has no attribute '{}'".format(name))


