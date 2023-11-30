from spider.elements.grid import OccupancyGrid2D

class GridTransformer:
    '''
    用于世界坐标系(局部笛卡尔坐标)和BEV下栅格坐标系的转换
    现在只支持2D
    '''
    def __init__(self):
        pass

    # def set_param(self, grid: OccupancyGrid2D):
    #     self.grid_resolution =
    #     self.ego_x_grid =
    #     self.ego_y_grid =


    def cart2grid(self, grid: OccupancyGrid2D, x_cart, y_cart, vx_cart=0, vy_cart=0):
        # 注意，grid中的x,y是图像坐标系下的，即图像中宽度上从左到右为x正方向，高度上从上到下为y正方向。尤其高度容易弄混。
        # grid坐标系，自车位置固定在grid中央某一位置不动（这个位置由grid对象中的lon_range和lat_range和grid_resolution决定）,自车车头在图像中始终朝上
        # C++文件中有个calc_ogm_idx什么的函数，可以借鉴

        if vx_cart == 0 and vy_cart == 0:
            vx_grid, vy_grid = 0, 0
            # 0阶坐标变换...
        else:
            # 0阶和1阶坐标变换。。。
            pass

        return x_grid, y_grid, vx_grid, vy_grid


    def grid2cart(self, grid: OccupancyGrid2D, x_grid, y_grid, vx_grid=0, vy_grid=0):

        if vx_grid == 0 and vy_grid == 0:
            vx_cart, vy_cart = 0, 0
            # 0阶坐标变换...
        else:
            # 0阶和1阶坐标变换。。。
            pass

        return x_cart, y_cart, vx_cart, vy_cart

