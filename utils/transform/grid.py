
import sys

# sys.path.append("D:/科研工作/自动驾驶决策/to康安(1)/")
from spider.elements.grid import OccupancyGrid2D
import numpy as np
from typing import List

'''
QKA完成
'''

class GridTransformer:
    '''
    用于世界坐标系(局部笛卡尔坐标)和BEV下栅格坐标系的转换
    现在只支持2D
    # 速度也要缩放吗？
    '''

    def __init__(self, ego: List[float] = [0, 0, 0, 0, 0]):

        # input:自车信息 list[1, 5] = [x, y, heading, vx, vy]
        # 自车heading角 - > pi/2-heading坐标旋转角
        self._angle = np.pi / 2 - ego[2]

        self._car = ego
        self._ego_x_cart, self._ego_y_cart = ego[0], ego[1]
        self._ego_vx_cart, self._ego_vy_cart = ego[-2], ego[-1]

    # def set_param(self, grid: OccupancyGrid2D):
    #     self.grid_resolution =
    #     self.ego_x_grid =
    #     self.ego_y_grid =
    @staticmethod
    def translation(x, y, _trans, vx=0, vy=0):
        '''平移变换
        0阶,1阶
        input args:
        x, y denotes 位置
        vx, vx denotes 速度

        output args:
        x_result, y_result, vx_result, vy_result 变换后位置和速度
        '''
        # trans_m = np.array([_trans[0], _trans[-1]]) # 平移矩阵 [x,y]
        d_result = np.array([x, y]) + _trans

        x_result, y_result = d_result[0], d_result[-1]

        if vx == 0 and vy == 0:
            vx_result, vy_result = 0, 0
        else:
            v_result = np.array([vx, vy]) + _trans
            vx_result, vy_result = v_result[0], v_result[-1]
        return x_result, y_result, vx_result, vy_result

    @staticmethod
    def rotation(x, y, angle, vx=0, vy=0):
        '''旋转变换
        0阶,1阶
        Args:
        input args:
        x, y denotes 位置
        angle: rota
        vx, vx denotes 速度

        output args:
        x_result, y_result, vx_result, vy_result 变换后位置和速度
        '''
        # _dtype = (
        #         x.dtype
        #     )
        rota_m = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],

        ],

        ).T  # 旋转矩阵
        # 坐标旋转变换
        d_result = np.array([x, y]) @ rota_m
        print('旋转后{}'.format(d_result))
        x_result, y_result = d_result[0], d_result[-1]

        if vx == 0 and vy == 0:
            vx_result, vy_result = 0, 0
        else:
            v_result = np.array([vx, vy]) @ rota_m
            vx_result, vy_result = v_result[0], v_result[-1]
        return x_result, y_result, vx_result, vy_result

    @staticmethod
    def scale(x, y, unit, vx=0, vy=0):
        '''缩放变换
        0阶,1阶
        input args:
        x, y denotes 位置
        vx, vx denotes 速度

        output args:
        x_result, y_result, vx_result, vy_result 变换后位置和速度
        '''
        # scale_m = np.array([unit, unit]) # 缩放量
        d_result = np.array([x, y]) / unit
        x_result, y_result = np.ceil(d_result[0]), np.ceil(d_result[-1])

        if vx == 0 and vy == 0:
            vx_result, vy_result = 0, 0
        else:
            v_result = np.array([vx, vy]) / unit
            vx_result, vy_result = np.ceil(v_result[0]), np.ceil(v_result[-1])
        return x_result, y_result, vx_result, vy_result

    @staticmethod
    def filp_x(x, y, vx=0, vy=0):
        """ 翻转变换
        0阶,1阶
        Args:
        x, y denotes 位置
        vx, vx denotes 速度

        Returns:
        x_result, y_result, vx_result, vy_result 变换后位置和速度
        """
        x_result, y_result = -x, y

        if vx == 0 and vy == 0:
            vx_result, vy_result = 0, 0
        else:
            vx_result, vy_result = -vx, vy
        return x_result, y_result, vx_result, vy_result

    @staticmethod
    def filp_y(x, y, vx=0, vy=0):
        """ 翻转变换
        0阶,1阶
        Args:
        x, y denotes 位置
        vx, vx denotes 速度

        Returns:
        x_result, y_result, vx_result, vy_result 变换后位置和速度
        """
        x_result, y_result = x, -y

        if vx == 0 and vy == 0:
            vx_result, vy_result = 0, 0
        else:
            vx_result, vy_result = vx, -vy
        return x_result, y_result, vx_result, vy_result

    def cart2occ(self, grid: OccupancyGrid2D, x_cart, y_cart, vx_cart=0, vy_cart=0):
        # 注意，grid中的x,y是图像坐标系下的，即图像中宽度上从左到右为x正方向，高度上从上到下为y正方向。尤其高度容易弄混。
        # grid坐标系，自车位置固定在grid中央某一位置不动（这个位置由grid对象中的lon_range和lat_range和grid_resolution决定）,自车车头在图像中始终朝上
        # C++文件中有个calc_ogm_idx什么的函数，可以借鉴
        """ cart -> occ
        Args:
        grid:OccupancyGrid2D
        x_cart, y_cart position in global
        vx_cart, vy_cart velocity in global

        returns:
        x_grid, y_grid position(pixel) in grid
        vx_grid, vy_grid position(pixel) in grid
        """
        # get ego points in grid coordination
        ego_cart = np.zeros((1, 2))  # 自定义原点与自车重合
        # 自定义自车世界坐标
        ego_x_cart = self._ego_x_cart
        ego_y_cart = self._ego_y_cart
        ego_vx_cart = self._ego_vx_cart
        ego_vy_cart = self._ego_vy_cart

        # grid
        ego_x_occ = grid.lateral_range[0]
        ego_y_occ = grid.longitudinal_range[0]
        # occ_resolution = grid.grid_resolution

        deltax_cart = x_cart - ego_x_cart
        deltay_cart = y_cart - ego_y_cart

        # 压缩科学计数
        np.set_printoptions(suppress=True)
        if vx_cart == 0 and vy_cart == 0:
            # 0阶坐标变换
            vx_occ, vy_occ = 0, 0
            # vx_pixel, vy_pixel = 0, 0
            # # 旋转
            deltax_occ, deltay_occ, _, _ = self.rotation(deltax_cart, deltay_cart, self._angle)
            # y翻转 -> delta in occ
            deltax_occ, deltay_occ, _, _ = self.filp_y(deltax_occ, deltay_occ)
            # 平移 -> x,y in occ
            x_occ, y_occ, _, _ = self.translation(deltax_occ, deltay_occ, np.array([ego_x_occ, ego_y_occ]))


        else:
            # 0阶和1阶坐标变换
            deltavx_cart = vx_cart - ego_vx_cart
            deltavy_cart = vy_cart - ego_vy_cart

            # 旋转
            deltax_occ, deltay_occ, deltavx_occ, deltavy_occ = self.rotation(deltax_cart, deltay_cart, self._angle,
                                                                             deltavx_cart, deltavy_cart)
            # y翻转 -> delta in occ
            deltax_occ, deltay_occ, deltavx_occ, deltavy_occ = self.filp_y(deltax_occ, deltay_occ, deltavx_occ,
                                                                           deltavy_occ)
            # 平移 -> x,y in occ
            x_occ, y_occ, _, _ = self.translation(deltax_occ, deltay_occ, np.array([ego_x_occ, ego_y_occ]))
            vx_occ, vy_occ = deltavx_occ + 0, deltavy_occ + 0

        return x_occ, y_occ, vx_occ, vy_occ

    def grid2occ(self, grid: OccupancyGrid2D, x_grid, y_grid, vx_grid=0, vy_grid=0):
        grid_resolution = np.array(grid.grid_resolution)
        # 缩放 -> grid to x,y
        x_occ, y_occ, vx_occ, vy_occ = self.scale(x_grid, y_grid, 1 / grid_resolution, vx_grid, vy_grid)
        return x_occ, y_occ, vx_occ, vy_occ

    # 速度也要缩放吗？
    def occ2grid(self, grid: OccupancyGrid2D, x_occ, y_occ, vx_occ=0, vy_occ=0):
        grid_resolution = grid.grid_resolution
        # 缩放 -> x,y to grid
        x_grid, y_grid, vx_grid, vy_grid = self.scale(x_occ, y_occ, grid_resolution, vx_occ, vy_occ)
        return x_grid, y_grid, vx_grid, vy_grid

    def occ2cart(self, grid: OccupancyGrid2D, x_occ, y_occ, vx_occ=0, vy_occ=0):
        """occ -> cart
        Args:
        grid:OccupancyGrid2D
        x_grid, y_grid position in grid (pixel)
        vx_grid, vy_grid velocity in grid (pixel)

        returns:
        x_cart, y_cart position in cart
        vx_cart, vy_cart position in cart


        """
        # get ego points in grid coordination
        ego_x_occ = grid.lateral_range[0]
        ego_y_occ = grid.longitudinal_range[0]
        grid_resolution = np.array(grid.grid_resolution)

        # 自定义自车世界坐标
        ego_x_cart = self._ego_x_cart
        ego_y_cart = self._ego_y_cart
        ego_vx_cart = self._ego_vx_cart
        ego_vy_cart = self._ego_vy_cart

        deltax_occ = x_occ - ego_x_occ
        deltay_occ = y_occ - ego_y_occ

        np.set_printoptions(suppress=True)

        if vx_occ == 0 and vy_occ == 0:
            vx_cart, vy_cart = 0, 0
            # 0阶坐标变换...
            deltax_occ, deltay_occ, _, _ = self.filp_y(deltax_occ, deltay_occ)

            deltax_cart, deltay_cart, _, _ = self.rotation(deltax_occ, deltay_occ, -self._angle)

            x_cart, y_cart, _, _ = self.translation(deltax_cart, deltay_cart, np.array([ego_x_cart, ego_y_cart]))
        else:
            # 0阶和1阶坐标变换
            deltavx_occ = vx_occ - 0
            deltavy_occ = vy_occ - 0

            deltax_occ, deltay_occ, deltavx_occ, deltavy_occ = self.filp_y(deltax_occ, deltay_occ, deltavx_occ,
                                                                           deltavy_occ)

            deltax_cart, deltay_cart, deltavx_cart, deltavy_cart = self.rotation(deltax_occ, deltay_occ, -self._angle,
                                                                                 deltavx_occ, deltavy_occ)

            x_cart, y_cart, _, _ = self.translation(deltax_cart, deltay_cart, np.array([ego_x_cart, ego_y_cart]))
            vx_cart, vy_cart = deltavx_cart + ego_vx_cart, deltavy_cart + ego_vy_cart

        return x_cart, y_cart, vx_cart, vy_cart


# TransMatrix.rotation(100, 30, np.pi)

if __name__ == '__main__':
    # 实例化OccupancyGrid2D
    grid = OccupancyGrid2D([100, 30], [30, 30], [0.1, 0.1], 4)
    # 实例化transformer
    gt = GridTransformer([0, 0, np.pi / 2, 0, 0])
    # 对(100, 100, 10, 10) transform
    # cart to occ
    x_occ, y_occ, vx_occ, vy_occ = gt.cart2occ(grid, 100, 100, 10, 10)
    print('世界(100,100)经过坐标变换后得到栅格坐标为:{} and {}'.format(x_occ, y_occ))
    # occ to grid
    x_grid, y_grid, vx_grid, vy_grid = gt.occ2grid(grid, x_occ, y_occ, vx_occ, vy_occ)

    # grid to occ
    x_occ, y_occ, vx_occ, vy_occ = gt.grid2occ(grid, x_grid, y_grid, vx_grid, vy_grid)

    # occ to cart
    x_cart, y_cart, vx_cart, vy_cart = gt.occ2cart(grid, x_occ, y_occ, vx_occ, vy_occ)
    print('栅格坐标经过坐标变换后得到世界坐标为:{} and {}'.format(x_cart, y_cart))

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

