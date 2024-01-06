from typing import Union, Type, Sequence
import matplotlib.pyplot as plt
import numpy as np

import spider.elements as elm
from spider.visualize.surface import draw_obb, draw_polygon



def draw_local_map(local_map: Union[elm.LocalMap,elm.RoutedLocalMap]):
    pass

def draw_lane(lane: elm.Lane):
    pass

def draw_ego_vehicle(ego_veh_state: elm.VehicleState, *args, fill=False, **kwargs):
    return draw_obb(ego_veh_state.obb, *args, fill=fill, **kwargs)

def draw_trackingbox_list():
    pass

def draw_boundingbox(bbox: elm.BoundingBox, *args, fill=False, **kwargs):
    return draw_polygon(bbox.vertices, *args, fill=fill, **kwargs)

def draw_history():
    pass

def draw_ego_history(ego_veh_state: elm.VehicleState, *args, clear:bool=False, **kwargs):
    # todo: 权宜之计，暂时在作画的时候就把自车轨迹记录下来，但是以后要规范化history的定义，然后在planner里面记录，
    #  然后再调用draw_history画
    if (not hasattr(draw_ego_history, "ego_history")) or clear:
        draw_ego_history.ego_history = [[], []]
    draw_ego_history.ego_history[0].append(ego_veh_state.x())
    draw_ego_history.ego_history[1].append(ego_veh_state.y())
    plt.plot(draw_ego_history.ego_history[0], draw_ego_history.ego_history[1], *args, **kwargs)

def draw_heatmap():
    pass

def draw_prediction():
    pass

def draw_path(path:elm.Path, *args, **kwargs):
    return draw_trajectory(path, *args, **kwargs)

def draw_trajectory(traj: Union[elm.Path, elm.Trajectory, elm.FrenetTrajectory], *args,
                    show_footprint=False, footprint_size=(5., 2.), footprint_fill=True, footprint_alpha=0.1,  **kwargs):
    lines = plt.plot(traj.x, traj.y,  *args, **kwargs)

    if show_footprint:
        length, width = footprint_size
        color = lines[0].get_color()
        footprint_alpha = footprint_alpha if footprint_fill else 0.8 # 填充就按设定的透明度来，否则默认0.8

        for x, y, yaw in zip(traj.x, traj.y, traj.heading):
            draw_obb((x, y, length, width, yaw), fill=footprint_fill, alpha=footprint_alpha, color=color)

    return lines


def draw_candidate_trajectories(candidate_trajectories:Sequence[elm.Trajectory],*args,
                                show_cost_colormap=False, cost=None,
                                show_footprint=False, **kwargs):
    '''
    candidate_trajectories默认不画脚印
    '''
    if show_cost_colormap:
        for traj in candidate_trajectories:
            draw_trajectory(traj, *args, show_footprint=show_footprint, **kwargs)
    else:
        raise NotImplementedError("To be completed...")  # todo:完善


def draw_trajectory_3d():
    '''
    z轴是时间
    '''
    pass

def draw_corridor_boxes_3d():
    pass



def show_occupancy():
    pass



