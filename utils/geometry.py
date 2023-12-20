import numpy as np
import math
from spider.elements.vector import project

def find_nearest_point(point: np.ndarray, target_points):
    x,y = point

    all_dist2 = (target_points[:, 0] - x) ** 2 + (target_points[:, 1] - y) ** 2

    nearest_idx = np.argmin(all_dist2)
    min_dist = math.sqrt(all_dist2[nearest_idx])
    return nearest_idx, min_dist

# def find_nearest_point(point: np.ndarray, target_points, roi=(-1,-1)):
#     """
#     roi:(x_dist_range,y_dist_range)感兴趣区域，(-1,-1)表示无感兴趣区域。表示的是猜测最近点的距离存在于当前点附近的x,y范围。
#         roi帮助加速算法
#     todo: 能不能加速算法？？得测试一下的。结论：不能加速，反而减速了！！切片操作和逻辑判断慢了
#     """
#     x,y = point
#     roi = np.array(roi)
#
#     if np.any(roi<0):
#         all_dist2 = (target_points[:, 0] - x) ** 2 + (target_points[:, 1] - y) ** 2
#
#     else:
#         x_range, y_range = roi
#         all_dist2 = np.ones(len(target_points)) * np.inf
#         roi_idx = (np.abs(target_points[:, 0] - x) < x_range) & (np.abs(target_points[:, 1] - y) < y_range)
#         roi_dist2 = (target_points[roi_idx, 0] - x) ** 2 + (target_points[roi_idx, 1] - y) ** 2
#         all_dist2[roi_idx] = roi_dist2
#
#     nearest_idx = np.argmin(all_dist2)
#     min_dist = math.sqrt(all_dist2[nearest_idx])
#     return nearest_idx, min_dist


def point_to_segment_distance(point: np.ndarray, segment_start:np.ndarray, segment_end:np.ndarray, allow_extension=True):
    """
    计算点到线段的最短距离
    allow_extension: 如果true,那么在投影点不在线段上时，允许投影点在线段延长线上来计算距离；如果false，不允许投影点在延长线上，
                        投影点强行拉回到线段端点，距离认为是端点到点的距离。
    """
    segment_vector = segment_end - segment_start
    point_vector = point - segment_start

    projection, distance_signed = project(point_vector, segment_vector, calc_distance=True)

    if not allow_extension:
        segment_length = np.linalg.norm(segment_vector)
        if projection<0:
            projection = 0.
            distance = np.linalg.norm(point_vector) # 到segment起点距离
            distance_signed = math.copysign(distance, distance_signed) # 附上符号
        elif projection > segment_length:
            projection = segment_length
            distance = np.linalg.norm(point-segment_end) # 到segment终点距离
            distance_signed = math.copysign(distance, distance_signed)  # 附上符号

    return projection, distance_signed # projection是点乘带正负；distance_signed是叉乘，如果是负，表示是顺时针，在右边

# def point_to_polyline_distance(point:np.ndarray, polyline:np.ndarray, accurate_method:bool=False):
#     """
#     qzl:
#     精确解法，即把每一个线段的距离都算出来，取最小，但效率低。
#     粗略解法，找距离最近的顶点，再在顶点附近找最近的线段。适用于折线的顶点密度足够大 或 折线弯曲程度不大的情形，一般的情形都有效
#     polyline: N行2列， x列 & y列， 表示折线段上所有的顶点
#     """
#     if accurate_method:
#         dists = [point_to_segment_distance(point, polyline[i], polyline[i+1])[1] for i in range(len(polyline)-1)]
#         minidx = np.argmin(np.abs(dists))
#         mindist = dists[minidx]
#         return mindist
#     else:
#         pass # todo: 写完，类似于笛卡尔转frenet
#
#
#     pass


if __name__ == '__main__':
    xs = np.linspace(0,200,200)
    ys = np.random.random(xs.shape[0])
    tgt_pts = np.column_stack((xs,ys))
    pt = np.array([50,0.5])
    roi=[5,5]

    import time
    t1 = time.time()
    for i in range(100000):
        nearest_idx, min_dist = find_nearest_point(pt,tgt_pts,roi)
    t2 = time.time()
    print("with roi:")
    print(t2-t1)
    print(nearest_idx, min_dist)

    t1 = time.time()
    for i in range(100000):
        nearest_idx, min_dist = find_nearest_point(pt, tgt_pts)
    t2 = time.time()
    print("without roi:")
    print(t2 - t1)
    print(nearest_idx, min_dist)
    # print(xs,ys)
