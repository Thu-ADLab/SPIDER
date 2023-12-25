import numpy as np
import math
import cv2

from spider.elements.Box import TrackingBoxList
import os

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

'''
qzl: 可以考虑直接继承ndarray类？还是.grid属性更好？
'''
class OccupancyGrid2D:
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

    @classmethod
    def from_grid(cls, grid: np.ndarray, grid_resolution, ego_anchor) -> 'OccupancyGrid2D':
        '''
        从栅格图像构建OccupancyGrid2D对象的类方法。

        Parameters:
        - grid: np.ndarray
            栅格图像，三维数组，形状为 (height, width, channel)。
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
        height, width, channel = grid.shape
        ego_x_occ, ego_y_occ = ego_anchor
        lon_range = [ego_y_occ * grid_resolution[0], (height - ego_y_occ) * grid_resolution[0]]
        lat_range = [ego_x_occ * grid_resolution[1], (width - ego_x_occ) * grid_resolution[1]]
        occ = cls(lon_range, lat_range, grid_resolution, channel)
        occ.set_grid(grid)
        return occ

    # @classmethod
    # def from_trackingboxlist(cls, trackingbox_list:TrackingBoxList,longitudinal_range, lateral_range,
    #                          grid_resolution, ego_anchor) -> 'OccupancyGrid2D':
    #     # 先初始化一个对应大小的图像【本质上是grid】
    #     # 从tb_list去逐个遍历bbox的vertices，每个判断是不是在range里面
    #     # 如果是，就把这个bbox画在图像上【这里涉及到坐标转换】
    #     # 如果不是就跳过
    #     # 至此，Grid的生成完成
    #     # 调用from_grid函数，得到occupancygrid2d类，然后返回
    #
    #     pass

    # qka完成
    # from spider.utils.transform.grid import GridTransformer
    @staticmethod
    def get_bbox_grid(gt: "GridTransformer", grid, bbox):

        x0y0_cart = bbox[0]
        x1y1_cart = bbox[1]

        x2y2_cart = bbox[2]
        x3y3_cart = bbox[3]

        x0_occ, y0_occ, vx0_occ, vy0_occ = gt.cart2occ(grid, x0y0_cart[0], x0y0_cart[-1])
        x1_occ, y1_occ, vx1_occ, vy1_occ = gt.cart2occ(grid, x1y1_cart[0], x1y1_cart[-1])
        x2_occ, y2_occ, vx2_occ, vy2_occ = gt.cart2occ(grid, x2y2_cart[0], x2y2_cart[-1])
        x3_occ, y3_occ, vx3_occ, vy3_occ = gt.cart2occ(grid, x3y3_cart[0], x3y3_cart[-1])

        x0_grid, y0_grid, vx0_occ, vy0_occ = gt.occ2grid(grid, x0_occ, y0_occ)
        x1_grid, y1_grid, vx1_grid, vy1_grid = gt.occ2grid(grid, x1_occ, y1_occ)
        x2_grid, y2_grid, vx2_grid, vv2_grid = gt.occ2grid(grid, x2_occ, y2_occ)
        x3_grid, y3_grid, vx3_grid, vy3_grid = gt.occ2grid(grid, x3_occ, y3_occ)

        pt0 = gt.occ2grid(grid, x0_occ, y0_occ)
        pt1 = gt.occ2grid(grid, x1_occ, y1_occ)
        pt2 = gt.occ2grid(grid, x2_occ, y2_occ)
        pt3 = gt.occ2grid(grid, x3_occ, y3_occ)

        pt = np.array([pt0, pt1, pt2, pt3])
        print(
            '第1个点的栅格坐标为:{:.6f},{:.6f}\n第2个点的栅格坐标为:{:.6f},{:.6f} \n第3个点的栅格坐标为:{:.6f},{:.6f} \n第4个点的栅格坐标为:{:.6f},{:.6f} \n '
            .format(x0_grid, y0_grid, x1_grid, y1_grid, x2_grid, y2_grid, x3_grid, y3_grid))
        return pt

    @classmethod
    def from_trackingboxlist_trans(cls, trackingbox_list: TrackingBoxList, grid_resolution,
                                   lon_range, lat_range, ego_anchor) -> 'OccupancyGrid2D':
        '''生成环境栅格, 具有旋转变换版本
        Args: trackingbox_list 障碍物对象的实例
        grid_resolution, lon_range, lat_range 图像初始化信息
        ego_anchor 固定的自车框坐标

        Returns: occ,OccupancyGrid2D 对象,具有环境栅格信息
        '''
        # 先初始化一个对应大小的图像[本质上是grid]
        # 从tb_list去逐个遍历bbox的vertices，每个判断是不是在range里面
        # #如果是，就把这个bbox画在图像上[这里涉及到坐标转换]
        # #如果不是就跳过
        # 至此，Grid的生成完成
        # 调用from_grid函数，得到occupancygrid2d类，然后返回

        # 初始化图像
        channel = 3
        grid = cls(lon_range, lat_range, grid_resolution, channel)

        bboxes_vertices = trackingbox_list.getBoxVertices()

        gt = GridTransformer([0, 0, np.pi / 3, 0, 0])
        image = (grid.get_occupancy() * 255).astype(np.uint8)
        for bbox in bboxes_vertices:
            pt = cls.get_bbox_grid(gt, grid, bbox)
            print('The transform point are:{}'.format(pt))
            print('The grid parameter(hw) are:{} and {}'.format(grid.height, grid.width))
            x0_grid, y0_grid = pt[0, 0:2]
            x1_grid, y1_grid = pt[1, 0:2]
            x2_grid, y2_grid = pt[2, 0:2]
            x3_grid, y3_grid = pt[3, 0:2]

            xmin_grid = np.min([x0_grid, x1_grid, x2_grid, x3_grid])
            xmax_grid = np.max([x0_grid, x1_grid, x2_grid, x3_grid])
            ymin_grid = np.min([y0_grid, y1_grid, y2_grid, y3_grid])
            ymax_grid = np.max([y0_grid, y1_grid, y2_grid, y3_grid])

            if xmin_grid < 0 or xmax_grid > grid.width or ymin_grid < 0 or ymax_grid > grid.height:
                continue
            else:
                print('该车在检测的栅格上')
                # grid.visual_rectangular( (grid.get_occupancy() * 255).astype(np.uint8), (xmax_grid, ymax_grid), (xmin_grid, ymin_grid) )
                pts = np.array([(x0_grid, y0_grid), (x1_grid, y1_grid), (x2_grid, y2_grid), (x3_grid, y3_grid)],
                               np.int32)
                cv2.rectangle(image, (int(xmax_grid), int(ymax_grid)), (int(xmin_grid), int(ymin_grid)), color=255,
                              thickness=2)
                # cv2.polylines(image, [pts], True, (255), 10)

                cv2.fillPoly(image, [pts], (255))
                cv2.line(image, (int(x0_grid), int(y0_grid)), (int(x1_grid), int(y1_grid)), color=255, thickness=1)
                cv2.line(image, (int(x1_grid), int(y1_grid)), (int(x2_grid), int(y2_grid)), color=255, thickness=1)
                cv2.line(image, (int(x2_grid), int(y2_grid)), (int(x3_grid), int(y3_grid)), color=255, thickness=1)
                cv2.line(image, (int(x0_grid), int(y0_grid)), (int(x3_grid), int(y3_grid)), color=255, thickness=1)
        cv2.imshow("rectangle", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        folder_path = './gird fig'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        new_im = cv2.imwrite(os.path.join(folder_path, 'occupancy grid 2D.png'), image)
        # 单通道
        image = np.array([image]).transpose(1, 2, 0)
        # print(np.array(image).shape)
        # 三通道
        image1 = cv2.imread(os.path.join(folder_path, 'occupancy grid 2D.png'))
        print(image.shape)
        print(image1.shape)

        occ = cls.from_grid(image, grid_resolution, ego_anchor)
        occ.set_grid(image)
        return occ

    @classmethod
    def from_trackingboxlist(cls, trackingbox_list: TrackingBoxList, grid_resolution,
                             lon_range, lat_range, ego_anchor) -> 'OccupancyGrid2D':
        '''生成环境栅格, 自车坐标系下的版本
        Args: trackingbox_list 障碍物对象的实例
        grid_resolution, lon_range, lat_range 图像初始化信息
        ego_anchor 固定的自车框坐标

        Returns: occ,OccupancyGrid2D 对象,具有环境栅格信息
        '''
        # 先初始化一个对应大小的图像[本质上是grid]
        # 从tb_list去逐个遍历bbox的vertices，每个判断是不是在range里面
        # #如果是，就把这个bbox画在图像上[这里涉及到坐标转换]
        # #如果不是就跳过
        # 至此，Grid的生成完成
        # 调用from_grid函数，得到occupancygrid2d类，然后返回

        # 初始化图像
        channel = 3
        grid = cls(lon_range, lat_range, grid_resolution, channel)

        bboxes_vertices = trackingbox_list.getBoxVertices()

        image = (grid.get_occupancy() * 255).astype(np.uint8)
        for bbox in bboxes_vertices:

            x0y0_cart = bbox[0]
            x1y1_cart = bbox[1]
            x2y2_cart = bbox[2]
            x3y3_cart = bbox[3]

            x0y0_grid = np.array([grid.lateral_range[0], grid.longitudinal_range[0]]) + np.multiply(x0y0_cart,
                                                                                                    np.array([1, -1]))
            x0y0_grid = x0y0_grid / grid.grid_resolution

            x1y1_grid = np.array([grid.lateral_range[0], grid.longitudinal_range[0]]) + np.multiply(x1y1_cart,
                                                                                                    np.array([1, -1]))
            x1y1_grid = x1y1_grid / grid.grid_resolution

            x2y2_grid = np.array([grid.lateral_range[0], grid.longitudinal_range[0]]) + np.multiply(x2y2_cart,
                                                                                                    np.array([1, -1]))
            x2y2_grid = x2y2_grid / grid.grid_resolution

            x3y3_grid = np.array([grid.lateral_range[0], grid.longitudinal_range[0]]) + np.multiply(x3y3_cart,
                                                                                                    np.array([1, -1]))
            x3y3_grid = x3y3_grid / grid.grid_resolution

            pt = np.array([x0y0_grid, x1y1_grid, x2y2_grid, x3y3_grid])

            print('The transform point are:{}'.format(pt))
            print('The grid parameter(hw) are:{} and {}'.format(grid.height, grid.width))

            x0_grid, y0_grid = pt[0, 0:2]
            x1_grid, y1_grid = pt[1, 0:2]
            x2_grid, y2_grid = pt[2, 0:2]
            x3_grid, y3_grid = pt[3, 0:2]

            xmin_grid = np.min([x0_grid, x1_grid, x2_grid, x3_grid])
            xmax_grid = np.max([x0_grid, x1_grid, x2_grid, x3_grid])
            ymin_grid = np.min([y0_grid, y1_grid, y2_grid, y3_grid])
            ymax_grid = np.max([y0_grid, y1_grid, y2_grid, y3_grid])

            if xmin_grid < 0 or xmax_grid > grid.width or ymin_grid < 0 or ymax_grid > grid.height:
                print('该车不在检测范围之内')
                continue
            else:
                print('该车在检测的栅格上')

                pts = np.array([(x0_grid, y0_grid), (x1_grid, y1_grid), (x2_grid, y2_grid), (x3_grid, y3_grid)],
                               np.int32)
                cv2.rectangle(image, (int(xmax_grid), int(ymax_grid)), (int(xmin_grid), int(ymin_grid)), color=255,
                              thickness=2)
                # cv2.polylines(image, [pts], True, (255), 10)

                cv2.fillPoly(image, [pts], (255))
                cv2.line(image, (int(x0_grid), int(y0_grid)), (int(x1_grid), int(y1_grid)), color=255, thickness=1)
                cv2.line(image, (int(x1_grid), int(y1_grid)), (int(x2_grid), int(y2_grid)), color=255, thickness=1)
                cv2.line(image, (int(x2_grid), int(y2_grid)), (int(x3_grid), int(y3_grid)), color=255, thickness=1)
                cv2.line(image, (int(x0_grid), int(y0_grid)), (int(x3_grid), int(y3_grid)), color=255, thickness=1)
        # cv2.imshow("rectangle", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # folder_path = './gird fig'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # new_im = cv2.imwrite(os.path.join(folder_path, 'occupancy grid 2D.png'), image)
        # # 单通道
        # image = np.array([image]).transpose(1, 2, 0)
        # # print(np.array(image).shape)
        # # 三通道
        # image1 = cv2.imread(os.path.join(folder_path, 'occupancy grid 2D.png'))
        # print('生成的栅格尺寸为:{}'.format(image.shape))
        # print(image1.shape)

        occ = cls.from_grid(image, grid_resolution, ego_anchor)
        occ.set_grid(image)
        return occ


class OccupancyGrid3D:
    def __init__(self):
        pass

if __name__ == '__main__':
    occ_grid = OccupancyGrid2D([100,30],[30,30],[0.1,0.1],4)
    occ_grid.visualize()


