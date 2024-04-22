
import sys
import math
import numpy as np
from typing import List

from spider.utils.transform.relative import RelativeTransformer


class GridTransformer:
    def __init__(self, longitudinal_range, lateral_range, grid_resolution,
                 ego_x=None, ego_y=None, ego_yaw=None, ego_vx=None, ego_vy=None):
        '''
        longitudinal_range: [longitudinal_range_front,longitudinal_range_back]
        lateral_range: [lateral_range_left, lateral_range_right]
        grid_resolution: [longitudinal_resolution, lateral_resolution]
        '''

        self.longitudinal_range = longitudinal_range
        self.lateral_range = lateral_range
        self.grid_resolution = grid_resolution
        self.lon_resolution, self.lat_resolution = grid_resolution

        self.width = int(math.ceil(sum(longitudinal_range) / self.lon_resolution))
        self.height = int(math.ceil(sum(lateral_range) / self.lat_resolution))

        self.ego_grid_pos = self._calc_ego_grid_pos() # [Float, Float]!

        self.rel_tf = RelativeTransformer(ego_x, ego_y, ego_yaw, ego_vx, ego_vy)

    def _calc_ego_grid_pos(self):
        '''
        计算自车在栅格坐标系下的位置
        '''
        ego_grid_x = int(self.longitudinal_range[1] / self.grid_resolution[0]) # rear range
        ego_grid_y = int(self.lateral_range[0] / self.grid_resolution[1]) # left range
        return ego_grid_x, ego_grid_y

    def set_ego_pose(self, ego_x, ego_y, ego_yaw):
        self.rel_tf.set_ego_pose(ego_x, ego_y, ego_yaw)

    def set_ego_velocity(self, ego_vx, ego_vy):
        self.rel_tf.set_ego_velocity(ego_vx, ego_vy)

    def cart2grid(self, x, y, yaw=None, vx=None, vy=None, ego_pose=None, ego_velocity=None):

        # 先把坐标转为车辆的相对坐标
        rel_x, rel_y, rel_yaw, rel_vx, rel_vy = self.rel_tf.abs2rel(x, y, yaw, vx, vy, ego_pose, ego_velocity)

        # 然后缩放和取整来使得相对坐标转为栅格坐标
        x_grid = int(rel_x / self.grid_resolution[0] + self.ego_grid_pos[0])
        y_grid = int(-rel_y / self.grid_resolution[1] + self.ego_grid_pos[1]) # *-1 because the y-axis in grid is reversed

        return x_grid, y_grid, rel_yaw, rel_vx, rel_vy

    def grid2cart(self, x_grid, y_grid, rel_yaw=None, rel_vx=None, rel_vy=None, ego_pose=None, ego_velocity=None):
        rel_x = (x_grid - self.ego_grid_pos[0]) * self.grid_resolution[0]
        rel_y = -(y_grid - self.ego_grid_pos[1]) * self.grid_resolution[1]# *-1 because the y-axis in grid is reversed

        x, y, yaw, vx, vy = self.rel_tf.rel2abs(rel_x, rel_y, rel_yaw, rel_vx, rel_vy, ego_pose, ego_velocity)

        return x, y, yaw, vx, vy

    def cart2grid4boxes(self):
        pass




if __name__ == '__main__':
    from spider.elements.grid import OccupancyGrid2D
    import cv2
    occ_grid = OccupancyGrid2D([50, 30], [30, 30], [0.1, 0.1], 1)

    grid_tf = GridTransformer([50, 30], [30, 30], [0.1, 0.1])
    grid_tf.set_ego_pose(50, 15, 0.)

    gx, gy = grid_tf.cart2grid(55, 0)[:2]
    # occ_grid.grid[0, x, y] = 1
    cv2.circle(occ_grid.grid[0], (gx, gy), 10, 1, -1)
    print(gx, gy)

    x,y = grid_tf.grid2cart(gx, gy)[:2]
    print(x, y)

    occ_grid.show(0, 0)



# class GridTransformer:
#     '''
#     用于世界坐标系(局部笛卡尔坐标)和BEV下栅格坐标系的转换
#     现在只支持2D
#     '''
#     def __init__(self):
#         pass
#
#     # def set_param(self, grid: OccupancyGrid2D):
#     #     self.grid_resolution =
#     #     self.ego_x_grid =
#     #     self.ego_y_grid =
#
#
#     def cart2grid(self, grid: OccupancyGrid2D, x_cart, y_cart, vx_cart=0, vy_cart=0):
#         # 注意，grid中的x,y是图像坐标系下的，即图像中宽度上从左到右为x正方向，高度上从上到下为y正方向。尤其高度容易弄混。
#         # grid坐标系，自车位置固定在grid中央某一位置不动（这个位置由grid对象中的lon_range和lat_range和grid_resolution决定）,自车车头在图像中始终朝上
#         # C++文件中有个calc_ogm_idx什么的函数，可以借鉴
#
#         if vx_cart == 0 and vy_cart == 0:
#             vx_grid, vy_grid = 0, 0
#             # 0阶坐标变换...
#         else:
#             # 0阶和1阶坐标变换。。。
#             pass
#
#         return x_grid, y_grid, vx_grid, vy_grid
#
#
#     def grid2cart(self, grid: OccupancyGrid2D, x_grid, y_grid, vx_grid=0, vy_grid=0):
#
#         if vx_grid == 0 and vy_grid == 0:
#             vx_cart, vy_cart = 0, 0
#             # 0阶坐标变换...
#         else:
#             # 0阶和1阶坐标变换。。。
#             pass
#
#         return x_cart, y_cart, vx_cart, vy_cart

