from typing import Union, Type, Sequence
import matplotlib.pyplot as plt
import numpy as np

import spider.elements as elm
from spider.elements.box import aabb2vertices
from spider.visualize.surface import draw_obb, draw_polygon
from spider.visualize.line import draw_polyline
from spider.visualize.common import ego_centric_view


########### presets ##############
def _update_default_kwargs(default_kwargs:dict, kwargs:dict):
    temp = default_kwargs.copy()
    temp.update(kwargs)
    return temp

_ego_veh_kwargs = {"color":'C0', "fill":True, "alpha":0.3, "linestyle":'-', "linewidth":1.5}
_traj_kwargs = {"color":'C2', "linestyle":'-', "marker":'.', "linewidth":1.5}
##################################
# , color = 'C0', fill = True, alpha = 0.3, linestyle = '-', linewidth = 1.5
# def draw_vehicle_state(vehicle_state):
#

def lazy_draw(ego_state, trackingbox_list, local_map, trajectory):
    draw_ego_vehicle(ego_state)
    draw_trackingbox_list(trackingbox_list)
    draw_local_map(local_map)
    draw_trajectory(trajectory)
    ego_centric_view(ego_state.x(), ego_state.y())


def draw_local_map(local_map: Union[elm.LocalMap,elm.RoutedLocalMap]):
    for lane in local_map.lanes:
        plt.plot(lane.centerline[:, 0], lane.centerline[:, 1], color='gray', linestyle='--', lw=1.5)  # 画地图


def draw_lane(lane: elm.Lane):
    pass

def draw_ego_vehicle(ego_veh_state: elm.VehicleState, *args, **kwargs):
    kwargs = _update_default_kwargs(_ego_veh_kwargs, kwargs)
    return draw_obb(ego_veh_state.obb, *args, **kwargs)

def draw_trackingbox_list(trackingbox_list, draw_prediction=True, draw_history=False, *args, **kwargs):
    for tb in trackingbox_list:
        draw_boundingbox(tb, color='black', fill=True, alpha=0.1, linestyle='-', linewidth=1.5)  # 画他车
        if draw_prediction and (tb.prediction is not None) and (len(tb.prediction) > 0):
            draw_polyline(tb.prediction[:,:2], show_buffer=True, buffer_dist=tb.width * 0.5, buffer_alpha=0.1,
                          color='C3')

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
                    show_footprint=True, footprint_size=(5., 2.), footprint_fill=True, footprint_alpha=0.3,
                    gradual_transparency=True, **kwargs):
    if len(args) == 0: # 避免参数冗余或错乱
        kwargs = _update_default_kwargs(_traj_kwargs, kwargs)
    lines = plt.plot(traj.x, traj.y,  *args, **kwargs)

    _min_face_alpha = 0.1
    _min_edge_alpha = 0.3
    if show_footprint:
        length, width = footprint_size
        color = lines[0].get_color()
        footprint_alpha = footprint_alpha if footprint_fill else 0.8 # 填充就按设定的透明度来，否则默认0.8

        alphas = np.linspace(footprint_alpha, _min_face_alpha, traj.steps) if gradual_transparency else [footprint_alpha] * traj.steps
        edge_alphas = np.linspace(0.8, _min_edge_alpha, traj.steps) if gradual_transparency else [footprint_alpha] * traj.steps
        for i, (x, y, yaw) in enumerate(zip(traj.x, traj.y, traj.heading)):
            draw_obb((x, y, length, width, yaw), fill=footprint_fill, color=color,
                     alpha=alphas[i], edge_alpha=edge_alphas[i])

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

def draw_corridor_3d():
    pass

def draw_corridor(corridor,*args, **kwargs):
    '''
    要求corridor的形式是corridor: [ [x1_min, y1_min, x1_max, y1_max],... ]
    '''
    for aabb in corridor:
        vertices = aabb2vertices(aabb)
        draw_polygon(vertices, *args, **kwargs)


def show_occupancy():
    pass



