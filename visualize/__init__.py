from spider.visualize.common import *
from spider.visualize.line import *
from spider.visualize.point import *
from spider.visualize.surface import *
from spider.visualize.surface3d import *
from spider.visualize.elements import *

import matplotlib.pyplot as plt

def __getattr__(name):
    return getattr(plt, name)
    # if name == 'show':
    #     return plt.show
    # elif name == 'savefig':
    #     return plt.savefig
    # elif name == 'close':
    #     return plt.close
    # else:
    #     import warnings
    #     warnings.warn("{} is actually from matplotlib.pyplot. Please try to import plt directly.".format(name))
    #     return getattr(plt, name)

