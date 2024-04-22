import warnings
from typing import Union
import numpy as np
import math
import spider
try:
    import cv2
except (ModuleNotFoundError, ImportError) as e:
    cv2 = spider._virtual_import("cv2", e)

from spider.elements.box import TrackingBoxList, TrackingBox, obb2vertices
import os

'''
qzl:
todo list:

2栅格的预测没写【optional】
3栅格占据的uncertainty没有考虑
4栅格的dtype如何规定，并且没有明确规定占据是0-1填充还是0-255填充。建议0-1填充，这样子uncertainty好考虑。
6以后再加入全局静态的occupancy，一般室内机器人或者停车场可以用。现在默认是跟着自车坐标系对齐的
'''

def image_scale(image):
    '''convert an array of any scale into 0-255 image(uint8)'''
    image = np.asarray(image)
    image = image - np.min(image)
    image = (image / np.max(image) * 255).astype(np.uint8)
    return image


class OccupancyGrid2D:
    # def __init__(self, height, width, num_channel, grid_resolution,):
    def __init__(self, longitudinal_range, lateral_range, grid_resolution, num_channel=1, channel_names=("occ",)):
        '''
        环境的栅格化表达
        longitudinal_range: [longitudinal_range_front,longitudinal_range_back]
        lateral_range: [lateral_range_left, lateral_range_right]
        grid_resolution: [longitudinal_resolution, lateral_resolution]
        num_channel: int
        num_channel 0必须储存占据信息。
        '''
        for x in longitudinal_range, lateral_range, grid_resolution:
            assert len(x) == 2
        self.longitudinal_range = longitudinal_range
        self.lateral_range = lateral_range
        self.grid_resolution = grid_resolution

        self.width = int(math.ceil( sum(longitudinal_range) / self.lon_resolution ))
        self.height = int(math.ceil( sum(lateral_range) / self.lat_resolution ))
        self.num_channel = num_channel
        if (channel_names is None) or (channel_names) != num_channel:
            warnings.warn("Number of channel names does not match number of channels")
        else:
            self.channel_names = channel_names

        self.grid: np.ndarray = np.zeros((self.num_channel, self.height, self.width)) # 很核心的一个属性，存储栅格信息

    def get_occupancy(self):
        return self.grid[0]

    def set_grid(self, grid: np.ndarray):
        assert grid.shape == (self.num_channel, self.height, self.width)
        self.grid = grid

    def show(self, channel:Union[int, str], delay=0, winname='vis'):
        if isinstance(channel, str):
            channel_id = self.channel_names.index(channel)
        elif isinstance(channel, int):
            channel_id = channel
        else:
            raise ValueError("Channel should be either int or str. For example, 0 or 'occ'")
        img = self.grid[channel_id]
        cv2.imshow(winname, img)
        cv2.waitKey(delay)
        # cv2.destroyAllWindows()

    @staticmethod# qka
    def visual_rectangular(image, pt1, pt2, color=255, thickness=-1, delay=0):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))

        print(pt1, pt2)
        # image = (self.get_occupancy() * 255).astype(np.uint8)
        cv2.rectangle(image, pt1, pt2, color, thickness=thickness)
        # cv2.rectangle(image, (100, 100), (300, 300), color = color, thickness = -1)

    @property
    def lon_resolution(self):
        return self.grid_resolution[0]

    @property
    def lat_resolution(self):
        return self.grid_resolution[1]


    def dilate(self, kernel_size, channel=0, iterations=1):
        '''膨胀操作'''
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.grid[channel] = cv2.dilate(self.grid[channel], kernel, iterations=iterations)
        # return self.grid[channel]

    def gaussian_blur(self, kernel_size, channel=0):
        '''高斯模糊'''
        self.grid[channel] = cv2.GaussianBlur(self.grid[channel], (kernel_size, kernel_size), 0)
        # return self.grid[channel]

    @classmethod
    def from_grid(cls, grid: np.ndarray, grid_resolution, ego_anchor, channel_names=None) -> 'OccupancyGrid2D':
        '''
        从栅格图像构建OccupancyGrid2D对象的类方法。

        Parameters:
        - grid: np.ndarray
            栅格图像，三维数组，形状为 (num_channel, height, width)。
        - grid_resolution: tuple
            栅格分辨率，包含两个元素，分别是纵向和横向的分辨率。
        - ego_anchor: tuple
            Ego车辆在栅格图像中的锚点坐标 (x, y)。

        Returns:
        - occ: OccupancyGrid2D
            从栅格图像构建的OccupancyGrid2D对象。
        '''
        # ego_anchor是ego veh在图像中的x,y坐标
        assert len(grid.shape) == 3
        assert len(grid_resolution) == 2
        num_channel, height, width = grid.shape
        ego_x_grid, ego_y_grid = ego_anchor
        lat_range = [ego_y_grid * grid_resolution[1], (height - ego_y_grid) * grid_resolution[1]]
        lon_range = [(width - ego_x_grid) * grid_resolution[0], ego_x_grid * grid_resolution[0]]
        occ = cls(lon_range, lat_range, grid_resolution, num_channel)
        occ.set_grid(grid)
        return occ



    @classmethod
    def from_trackingboxlist(cls, trackingbox_list: TrackingBoxList, lon_range, lat_range, grid_resolution,
                             ego_pose, ego_velocity=(0.,0.), features=('occ','vx','vy')) -> 'OccupancyGrid2D':
        '''生成环境栅格, 自车坐标系下的版本
        Args: trackingbox_list 障碍物对象的实例
        grid_resolution, lon_range, lat_range 图像初始化信息
        features: ('occ','vx','vy') 代表写入栅格的特征
        ego_pose: (x, y, yaw) of ego
        ego_velocity: (vx, vy) of ego

        Returns: occ,OccupancyGrid2D 对象,具有环境栅格信息
        '''
        from spider.utils.transform.grid import GridTransformer

        # initialize grid
        num_channels = len(features)
        grid = cls(lon_range, lat_range, grid_resolution, num_channels, channel_names=features)

        # initialize grid trans
        ego_velocity = () if ego_velocity is None else ego_velocity
        grid_tf = GridTransformer(lon_range, lat_range, grid_resolution, *ego_pose, *ego_velocity)

        polygons = []
        vs = []

        assert grid_resolution[0] == grid_resolution[1], "Only support grid with the same scale for now..."
        for bbox in trackingbox_list:
            bbox: TrackingBox
            x, y, l, w, yaw = bbox.obb
            vx, vy = bbox.vx, bbox.vy
            gx, gy, gyaw, gvx, gvy = grid_tf.cart2grid(x, y, yaw, vx, vy) # todo: 这里有冗余计算, vx, vy
            gl, gw = l / grid_resolution[0], w / grid_resolution[1] # todo: 这个是错的。应该每个顶点算出来然后再转换坐标
            polygons.append(obb2vertices([gx, gy, gl, gw, gyaw]))
            vs.append([vx, vy])

        polygons = np.array(polygons, dtype=np.int32)
        vs = np.array(vs, dtype=np.float32)
        if 'occ' in features:
            channel_i = features.index('occ')
            cv2.fillPoly(grid.grid[channel_i], polygons, color=1)
            # cv2.polylines(grid.grid[channel_i], polygons, color=1, isClosed=True)
        if 'vx' in features:
            channel_i = features.index('vx')
            for j, poly in enumerate(polygons):
                cv2.fillPoly(grid.grid[channel_i], [poly], color=float(vs[j,0]))
                # cv2.polylines(grid.grid[channel_i], [poly], color=float(vs[j,0]), isClosed=True, thickness=-1)
        if 'vy' in features:
            channel_i = features.index('vy')
            for j, poly in enumerate(polygons):
                cv2.fillPoly(grid.grid[channel_i], [poly], color=float(vs[j, 0]))
                # cv2.polylines(grid.grid[channel_i], [poly], color=float(vs[j,1]), isClosed=True, thickness=-1)
        return grid


class OccupancyGrid3D:
    def __init__(self):
        pass

if __name__ == '__main__':
    # occ_grid = OccupancyGrid2D([100,30],[30,30],[0.1,0.1],4)
    # occ_grid.show(0,0)
    from spider.interface.BaseBenchmark import DummyBenchmark
    ego_state, tblist = DummyBenchmark.get_environment_presets()[:2]
    ego_pose = [ego_state.x(), ego_state.y(), ego_state.yaw()]
    occ_grid = OccupancyGrid2D.from_trackingboxlist(tblist, [100, 30], [30, 30],
                                                    [0.1, 0.1], ego_pose)
    occ_grid.dilate(5, 0, 5)
    occ_grid.gaussian_blur(9,0)
    occ_grid.show(0, 0)
    occ_grid.show(1, 0)

