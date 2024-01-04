import matplotlib.pyplot as plt
import numpy as np
from typing import Union

import spider
from spider.elements.trajectory import Trajectory, FrenetTrajectory
from spider.visualize.surface import draw_obb, draw_polygon
from spider.utils.geometry import generate_parallel_line


def draw_polyline(polyline:np.ndarray, *args, show_buffer=False, buffer_dist=1.0, buffer_alpha=0.2, **kwargs):
    polyline = np.asarray(polyline)
    lines = plt.plot(polyline[:,0], polyline[:,1], *args, **kwargs)

    if show_buffer:
        color = lines[0].get_color()
        left_bound = generate_parallel_line(polyline, buffer_dist, left_or_right=spider.DIRECTION_LEFT)
        right_bound = generate_parallel_line(polyline, buffer_dist, left_or_right=spider.DIRECTION_RIGHT)
        polygon_vertices = np.concatenate((left_bound, np.flip(right_bound, axis=0)))
        draw_polygon(polygon_vertices, fill=True, color=color, alpha=buffer_alpha)
    return lines


def draw_trajectory(traj: Union[Trajectory, FrenetTrajectory], *args,
                    show_footprint=False, footprint_size=(5., 2.), footprint_fill=True, footprint_alpha=0.1,  **kwargs):
    lines = plt.plot(traj.x, traj.y, *args, **kwargs)

    if show_footprint:
        length, width = footprint_size
        color = lines[0].get_color()
        footprint_alpha = footprint_alpha if footprint_fill else 0.8 # 填充就按设定的透明度来，否则默认0.8

        for x, y, yaw in zip(traj.x, traj.y, traj.heading):
            draw_obb((x, y, length, width, yaw), fill=footprint_fill, alpha=footprint_alpha, color=color)

    return lines


if __name__ == '__main__':
    pass


