import numpy as np
import math
import cv2

from spider.elements.Box import TrackingBoxList

'''
qzl:
todo list:

0缺少from_trackingboxlist方法和visualize方法
1栅格的碰撞检测没写
2栅格的预测没写【optional】
3栅格占据的uncertainty没有考虑
4栅格的dtype如何规定，并且没有明确规定占据是0-1填充还是0-255填充。建议0-1填充，这样子uncertainty好考虑。
5加入flag检索对应grid通道的功能。比如在init时候利用flag预先规定好grid包含[occupancy,class,vx,vy],
 并且能用flag进行增删改查操作，比如加一个heading通道，删去class通道，get和set occupancy通道。
6以后再加入全局静态的occupancy，一般室内机器人或者停车场可以用。现在默认是跟着自车坐标系对齐的
'''

class OccupancyGrid2D:
    '''
    qzl: 可以考虑直接继承ndarray类？还是.grid属性更好？
    '''
    # def __init__(self, height, width, channel, grid_resolution,):
    def __init__(self, longitudinal_range, lateral_range, grid_resolution, channel:int):
        '''
        环境的栅格化表达
        longitudinal_range: [longitudinal_range_front,longitudinal_range_back]
        lateral_range: [lateral_range_left, lateral_range_right]
        grid_resolution: [longitudinal_resolution, lateral_resolution]
        channel: int
        channel 0必须储存占据信息。
        '''
        for x in longitudinal_range, lateral_range, grid_resolution:
            assert len(x) == 2
        self.longitudinal_range = longitudinal_range
        self.lateral_range = lateral_range
        self.grid_resolution = grid_resolution

        self.height = int(math.ceil( sum(longitudinal_range) / self.lon_resolution ))
        self.width = int(math.ceil( sum(lateral_range) / self.lat_resolution ))
        self.channel = channel

        self.grid: np.ndarray = np.zeros((self.height, self.width, self.channel))

    def get_occupancy(self):
        return self.grid[:,:,0]

    def set_grid(self, grid: np.ndarray):
        assert grid.shape == (self.height, self.width, self.channel)
        self.grid = grid

    def visualize(self, delay=0):
        occ = (self.get_occupancy() * 255).astype(np.uint8)
        cv2.imshow('vis', occ)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

    @property
    def lon_resolution(self):
        return self.grid_resolution[0]

    @property
    def lat_resolution(self):
        return self.grid_resolution[1]

    @classmethod
    def from_grid(cls, grid: np.ndarray, grid_resolution, ego_anchor) -> 'OccupancyGrid2D':
        # ego_anchor是ego veh在图像中的x,y坐标
        assert len(grid.shape) == 3
        assert len(grid_resolution) == 2
        height, width, channel = grid.shape
        ego_x_occ, ego_y_occ = ego_anchor
        lon_range = [ego_y_occ * grid_resolution[0], (height - ego_y_occ) * grid_resolution[0]]
        lat_range = [ego_x_occ * grid_resolution[1], (width - ego_x_occ) * grid_resolution[1]]
        occ = cls(lon_range, lat_range, grid_resolution, channel)
        occ.set_grid(grid)
        return occ

    @classmethod
    def from_trackingboxlist(cls, trackingbox_list:TrackingBoxList,longitudinal_range, lateral_range,
                             grid_resolution, ego_anchor) -> 'OccupancyGrid2D':
        # 先初始化一个对应大小的图像【本质上是grid】
        # 从tb_list去逐个遍历bbox的vertices，每个判断是不是在range里面
        # 如果是，就把这个bbox画在图像上【这里涉及到坐标转换】
        # 如果不是就跳过
        # 至此，Grid的生成完成
        # 调用from_grid函数，得到occupancygrid2d类，然后返回

        pass


class OccupancyGrid3D:
    def __init__(self):
        pass

if __name__ == '__main__':
    occ_grid = OccupancyGrid2D([100,30],[30,30],[0.1,0.1],4)
    occ_grid.visualize()


