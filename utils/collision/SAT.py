from spider.elements.vector import *
import numpy as np
from .AABB import AABB_check


def SAT_check(vertices1:np.ndarray ,vertices2:np.ndarray):
    '''
    Separating Axis Theorem，SAT,分离轴定理，用于检测凸多边形碰撞
    TODO:qzl:两个矩形的话只需要检查4条边对应的轴（因为有平行），目前是默认8条全部检测。可以尝试用集合保证唯一性。后期可以改进。
    :param vertices1: 多边形1
    :param vertices2: 多边形2
    :return: 碰撞与否
    '''
    if not AABB_check(vertices1,vertices2):
        # 先粗检
        return False

    separating_axis_vec = []
    # TODO:qzl:下面的内容写成矩阵计算形式更快
    # 获取所有边的单位向量并储存
    for polygon in [vertices1,vertices2]:
        for i in range(len(polygon)):
            v1 = polygon[i]
            if i == len(polygon)-1:
                v2 = polygon[0]
            else:
                v2 = polygon[i+1]
            vertical_vec = rotate90(v2-v1)
            separating_axis_vec.append(vertical_vec)

    # TODO: qzl:投影的时候其实也不用单位化法向量
    # 计算每个顶点向量在分离轴上的投影，并检查是否重叠
    collision = True
    for axis_vec in separating_axis_vec:
        proj1 = project(vertices1, axis_vec)
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = project(vertices2, axis_vec)
        min2, max2 = np.min(proj2), np.max(proj2)
        if min1 > max2 or min2 > max1: # 说明没有重叠, 也就是存在分割线
            collision = False
            break

    return collision
