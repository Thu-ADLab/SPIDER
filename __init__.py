import importlib as _importlib
from spider.param import *
from spider._virtual_import import _virtual_import, _try_import, _try_import_from
from spider._misc import *
from spider.teaser import teaser


submodules = [
    'elements', 'utils', 'sampler', 'evaluator', 'interface', 'control', 'planner_zoo', 'RL',
    'data',
    #'teaser', 'param'
]

__all__ = submodules + ['teaser', 'param']


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'spider.{name}')
    elif name =="vehicle_model":
        return AttributeError("sub module 'spider.vehicle_model' has been moved to 'spider.control.vehicle_model'")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'spider' has no attribute '{name}'"
            )
