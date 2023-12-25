import numpy as np
from spider.elements.trajectory import Path
from scipy.spatial import cKDTree


class hybridAStarPlanner:
    def __init__(self, x_reso, y_reso, yaw_reso):
        self.x_reso = x_reso
        self.y_reso = y_reso
        self.yaw_reso = yaw_reso

    # def planWithCostMap(self, start_pose, end_pose, costmap):
    #     path = Path()
    #
    #     return path



if __name__ == '__main__':
    XY_GRID_RESO = 0.1
    YAW_GRID_RESO = np.deg2rad(15.)

    pass