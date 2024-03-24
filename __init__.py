import importlib as _importlib
from spider.param import *
from spider._virtual_import import _virtual_import, _try_import, _try_import_from
# from spider import constraints
# from spider import elements, utils, sampler, evaluator, interface, vehicle_model
# from spider import motion_planning, path_planning
# from spider import planner_zoo
# from spider import RL
from spider.teaser import teaser


submodules = [
    'elements', 'utils', 'sampler', 'evaluator', 'interface', 'vehicle_model', 'planner_zoo', 'RL',
    #'teaser', 'param'
]

__all__ = submodules + ['teaser', 'param']


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'spider.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'spider' has no attribute '{name}'"
            )
