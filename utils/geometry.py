import numpy as np
import math
from spider.elements.vector import project, rotate90
import spider


def resample_polyline(line, resolution):
    """
    Dense a polyline by linear interpolation.
    线性插值重新采样曲线上的点

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """
    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    s = np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0],s])
    num = int(round(s[-1]/resolution))

    try:
        s_space = np.linspace(0,s[-1],num = num)
    except:
        raise ValueError(num, s[-1], len(s))

    x = np.interp(s_space,s,line[:,0])
    y = np.interp(s_space,s,line[:,1])

    return np.array([x,y]).T

def cumulated_distances(polyline:np.ndarray):
    """
    Calculate the cumulated distances along a polyline
    :param polyline: array([[x0,y0], [x1,y1], ...])
    :return:
    """
    polyline = np.asarray(polyline)
    diff = np.diff(polyline, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    return np.insert(np.cumsum(dist), 0, 0.0)


def generate_parallel_line(polyline:np.ndarray, dist, left_or_right=spider.DIRECTION_LEFT):
    # todo: 看看需不需要修改，尤其是转化为矢量运算，现在可能效率比较低
    assert left_or_right in [spider.DIRECTION_LEFT, spider.DIRECTION_RIGHT]
    direction_sign = -1 if left_or_right == spider.DIRECTION_LEFT else 1

    polyline = np.asarray(polyline)
    parallel_line = np.zeros_like(polyline)

    for j, pt2 in enumerate(polyline):
        if j == 0:
            pt1 = polyline[j + 1]
            vec_long = pt1 - pt2
        else:
            pt1 = polyline[j - 1]
            vec_long = pt2 - pt1
        e_long = vec_long/np.linalg.norm(vec_long) # vector/its norm
        e_lat = direction_sign * rotate90(e_long)
        vec_lat = e_lat * dist
        parallel_line[j] = pt2 + vec_lat

    return parallel_line


def find_nearest_point(point, target_points):
    x, y = point

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

# def smooth_path(path, window_size=5):
#     """
#     平滑路径函数
#
#     参数:
#         path: numpy.ndarray, shape为 (N, 2)，包含路径中心点的坐标
#         window_size: int, 窗口大小，用于计算平均值，默认为5
#
#     返回:
#         smoothed_path: numpy.ndarray, 平滑后的路径，与输入路径形状相同
#     """
#     smoothed_path = np.copy(path)  # 创建一个新数组，以保留原始路径数据
#
#     # 遍历路径中的每个点
#     for i in range(len(path)):
#         # 确定窗口范围
#         start = max(0, i - window_size // 2)
#         end = min(len(path), i + window_size // 2 + 1)
#
#         # 计算窗口内点的平均值
#         window_slice = path[start:end, :]
#         smoothed_path[i] = np.mean(window_slice, axis=0)
#
#     return smoothed_path


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
