from spider.elements.box import TrackingBoxList, TrackingBox, BoundingBox
from spider.elements.map import ScenarioType, TrafficLight, Lane, LocalMap, RoutedLocalMap
from spider.elements.grid import OccupancyGrid2D
from spider.elements.vehicle import VehicleState, Location, Rotation, Transform
from spider.elements.vector import Vector3D
from spider.elements.trajectory import Trajectory, FrenetTrajectory, Path

from typing import Tuple, Union

Observation = Tuple[
    VehicleState,
    Union[TrackingBoxList,OccupancyGrid2D],
    Union[RoutedLocalMap,LocalMap]
]

Plan = Union[Trajectory, FrenetTrajectory] # will add control in the future version

def __getattr__(name):
    if name == "Box":
        raise ValueError("spider.elements.Box has been renamed as spider.elements.box (small case).")

    else:
        raise AttributeError("module 'spider.elements' has no attribute '{}'".format(name))


