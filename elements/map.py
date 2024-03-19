from typing import List, Sequence, Union
import enum
import numpy as np
import math

import spider
from spider.elements.curves import ParametricCubicSpline
from spider.elements.vehicle import VehicleState
from spider.utils import transform
from spider.utils.geometry import resample_polyline, generate_parallel_line

"""
当前场景：flag，Multi-lane / Junction
车道信息：车道列表，每条车道表示为Lane类，包含车道中心线(点序列)、车道边界线(点序列)、车道边界类型(实线/虚线/路沿)、车道宽度、车辆限速
导航信息：目标车道序号列表、到最迟换道点的距离
信号灯信息：前方路口信号灯状态、信号灯相位具体信息
交通标志信息：停车让行标志、禁止停车标志等。

"""

class ScenarioType(enum.Enum):
    Multilane = 0
    Junction = 1
    Parkinglot = 2

class TrafficLight(enum.Enum):
    Green = 0
    Red = 1
    Yellow = 2


class Lane:
    def __init__(self, index, centerline:np.ndarray, width=3.5, speed_limit=60/3.6, resample:bool=False, resample_resolution=1.0):
        """
        centerline是2列N行的ndarray，N无长度限制
        """
        self.id = index  # 左小右大，0，1，2，3
        self.virtual = False # 是否是虚拟车道（比如路口的虚拟reference line形成的虚拟车道）

        if resample:
            centerline = resample_polyline(centerline, resample_resolution)
        self.centerline = np.array(centerline)
        self.centerline_csp = ParametricCubicSpline(self.centerline[:, 0], self.centerline[:, 1])
        self.width = width
        self.speed_limit = speed_limit

        # self.centerline_csp = None # cubic spline for centerline

        self._left_laneline = None
        self._right_laneline = None

        self.left_lane_change = True
        self.right_lane_change = True # 是否允许左换道/右换道

        # qzl: maybe useful in FUTURE VERSIONS below
        self.bidirectional = False
        self.traffic_flow = 0
        self.traffic_density = 0
        self.average_speed = 0 # qzl:这三个都是交通特性，动态地图可以给出；详见greenshields模型


    # def densify(self, ds=1.0): # 在geometry里面
    #     pass

    def smoothen(self):
        pass

    @property
    def left_laneline(self):
        if self._left_laneline is None:
            self._left_laneline = generate_parallel_line(self.centerline, self.width/2.0, spider.DIRECTION_LEFT)
        return self._left_laneline

    @property
    def right_laneline(self):
        if self._right_laneline is None:
            self._right_laneline = generate_parallel_line(self.centerline, self.width / 2.0, spider.DIRECTION_RIGHT)
        return self._right_laneline


class LocalMap:
    def __init__(self, lanes:Sequence[Lane]=(), scene_type=ScenarioType.Multilane, section_id=-1,
                 network=None, traffic_signs=None):

        self.type = scene_type
        self.section_id = section_id
        self.lanes = list(lanes) # 这里指的是自车可以走的lanes

        self.network = network # qzl:FUTURE VERSION

        self.traffic_signs = traffic_signs # qzl:FUTURE VERSION

    def set_scenario_type(self, scenario_type):
        self.type = scenario_type

    def add_lane(self,lane:Lane):
        self.lanes.append(lane)

    # def set
    def get_centerline_info(self, lane_index):
        return self.lanes[lane_index].centerline, self.lanes[lane_index].centerline_csp

    def match_lane(self, ego_veh_state: VehicleState):
        if len(self.lanes) == 0:
            raise ValueError("No lanes!")

        x, y = ego_veh_state.x(), ego_veh_state.y()
        min_idx, min_dist = -1, math.inf
        coordinate_transformer = transform.FrenetCoordinateTransformer()
        for idx in range(len(self.lanes)):
            target_lane = self.lanes[idx]
            coordinate_transformer.set_reference_line(target_lane.centerline, target_lane.centerline_csp)
            fstate = coordinate_transformer.cart2frenet(x, y, order=0)
            dist = math.fabs(fstate.l)
            if dist < min_dist:
                min_idx, min_dist = idx, dist
        return min_idx

    @classmethod
    def from_centerlines(cls, centerline_array_list: List[np.ndarray],
                         lane_width:Union[Sequence, float] = 3.5,
                         speed_limit:Union[Sequence, float] = 60/3.6,
                         lane_idxs:Sequence = None,
                         resample: bool = False,
                         resample_resolution=1.0,
                         **lmap_kwargs):
        '''
        centerline_array_list包含N个centerline_array
        lane_width & speed_limit可以是常数，也可以是N个值的序列
        '''
        num = len(centerline_array_list)
        if lane_idxs is None:
            lane_idxs = np.arange(num)

        if not isinstance(lane_width, Sequence):
            lane_width = [lane_width] * num
        if not isinstance(speed_limit, Sequence):
            speed_limit = [speed_limit] * num


        lanes = []
        for id, centerline_arr, wid, v_lim in \
                zip(lane_idxs, centerline_array_list, lane_width, speed_limit):
            lanes.append(Lane(id, centerline_arr, wid, v_lim, resample, resample_resolution))

        return cls(lanes, **lmap_kwargs)




class RoutedLocalMap(LocalMap):
    def __init__(self, lanes:Sequence[Lane]=(), scene_type=ScenarioType.Multilane, section_id=-1,
                 network=None, traffic_signs=None):
        super(RoutedLocalMap, self).__init__(lanes, scene_type, section_id, network, traffic_signs)
        # 全局导航信息，暂时用不到
        self.route = None
        # 局部导航信息
        self.exit_lanes_idx = [] # 目标车道的序号集合
        self.distance_to_critical_point = 0.0  # todo:这个按道理需要结合自车定位特征，应该放到外面
        self.traffic_light_state = TrafficLight.Green

# if __name__ == '__main__':
#     def a(x:LaneLine):
#         print("okkk")
#
#     xx = LaneLine()