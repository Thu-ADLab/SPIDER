import numpy as np
'''
abandoned
'''

from .box import TrackingBoxList


class pointCloud:
    def __init__(self, ):
        pass

class globalOGM(np.ndarray):
    def __init__(self, size, channels, x_reso, y_reso, dtype=np.uint8):
        super(globalOGM, self).__init__((channels, size[0],size[1]), dtype)
        self.x_reso = x_reso
        self.y_reso = y_reso
        self.channels = channels
        self.height, self.width = size


class localOGM(np.ndarray):
    def __init__(self, size, channel, dtype):
        super(localOGM, self).__init__((channel, size[0], size[1]), dtype)
        pass

# def