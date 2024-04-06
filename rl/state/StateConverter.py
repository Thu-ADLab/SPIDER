import math
from typing import Union, List

import torch
import torch.nn as nn

import spider.elements as elm
import spider.rl.convert as cvt


# def to_obj_state(ego_veh_state:elm.VehicleState,
#                  perception:Union[elm.TrackingBoxList,elm.OccupancyGrid2D],
#                  local_map:Union[elm.RoutedLocalMap, elm.LocalMap],
#                  num_object = 10,
#                  num_waypoint = 50,
#                  x_range=(-80,80),  # if absolute=False, then it means the longitudinal range
#                  y_range=(-80,80),  # if absolute=False, then it means the lateral range
#                  speed_range=(-10,10),
#                  max_size=10,
#                  normalize=True,
#                  absolute=False,
#                  flatten=True,
#                  device=None,
#                  feature_keys:List[str]=None  # 目前暂时不起效，未来改进
#                  # 应该允许更多输入
#                  ):
#     """
#     # veh_info: [[presence, x, y, length, width, yaw, vx, vy] * Nv]
#     # refer_line: [[x, y] * Nw]
#     """
#     ego_info = [
#         1.0, *ego_veh_state.obb, ego_veh_state.velocity.x, ego_veh_state.velocity.y
#     ]
#
#     assert isinstance(perception, elm.TrackingBoxList), "Only TrackingBoxList is supported for object state"
#     all_obj_info = [ego_info]
#     num_valid_veh = min([len(perception), num_object - 1])
#     for i in range(num_valid_veh):
#         tbox = perception[i]
#         all_obj_info.append([1.0, *tbox.obb, tbox.vx, tbox.vy])
#
#
#     states = torch.tensor(all_obj_info, torch.float)
#     if not absolute: # 相对的运动学信息；
#         rel_tf = RelativeCoordinateTransformer(ego_info[1], ego_info[2], ego_info[5], ego_info[6], ego_info[7])
#         for i in range(1, num_valid_veh):
#             x,y,yaw,vx,vy= states[i][[1,2,5,6,7]]
#             x,y,yaw,vx,vy = rel_tf.abs2rel(x,y,yaw,vx,vy)
#             states[i][[1, 2, 5, 6, 7]] = torch.tensor([x,y,yaw,vx,vy])
#     if normalize:
#         states[:, 1] = normalize_to_range(states[:, 1], x_range[0], x_range[1])
#         states[:, 2] = normalize_to_range(states[:, 2], y_range[0], y_range[1])
#         states[:, 3:5] = normalize_to_range(states[:, 3:5], 0.0, max_size)
#         states[:, 5] = normalize_to_range(states[:, 5], -math.pi, math.pi)
#         states[:, 6:8] = normalize_to_range(states[:, 6:8], speed_range[0], speed_range[1])
#
#     # padding to fixed number of vehicles
#     # 最后再padding，以免参与了normalize和absolute的计算
#     delta_obj_num = num_object - len(all_obj_info)
#     if delta_obj_num > 0 :
#         states = pad_at_end(states, 0, delta_obj_num, 0.0)
#         # padding = (# padding的设置比较诡异。
#         #     0, 0, # 前两个数，表示最后一维的padding维数，前后各pad几维
#         #     0, delta_obj_num # 3-4个数，表示倒数第二维的padding维数，前后各pad几维
#         # )
#         # states = torch.nn.functional.pad(states, padding)
#
#
#     # concat当前地图信息
#     # refer_line = []
#
#
#     if flatten:
#         states = states.flatten()
#     if device is not None:
#         states = states.to(device)
#
#     return states


# class StateConverter:
#     def __init__(self):
#         pass
#
#     def encode(self, ego_veh_state:elm.VehicleState,
#                perception:Union[elm.TrackingBoxList,elm.OccupancyGrid2D],
#                local_map:Union[elm.RoutedLocalMap, elm.LocalMap]) -> torch.Tensor:
#         pass
#
#     def decode(self, state) -> elm.Observation:
#         pass


class KineStateEncoder(nn.Module):
    def __init__(self, num_object=10, # todo: 现在仅支持固定数量的，以后加一个支持不同长度的
                 num_waypoint=50,
                 relative=True,
                 feature_keys=None,  # todo:目前暂时不起效，未来改进
                 normalize=True, # 此参数以下都是normalize=True时才生效的参数
                 clamp=False,
                 x_range=(-80, 80),  # if relative=True, then it means the longitudinal range
                 y_range=(-80, 80),  # if relative=True, then it means the lateral range
                 speed_range=(-10, 10),
                 max_size=10,
                 ):
        super().__init__()

        self.state_dim = 8 * num_object # + 2 * num_waypoint

        if normalize:
            # 以后idx_range要用feature_keys自动生成索引
            idx_range = {1: x_range, 2: y_range, 3: (0., max_size), 4: (0., max_size),
                         5: (-math.pi, math.pi), 6: speed_range, 7: speed_range}
            self.encoder = cvt.Compose(
                cvt.ToKineState(num_object,num_waypoint, relative=relative),
                cvt.Normalize(idx_range_dict=idx_range, clamp=clamp),
                cvt.FixLength(0, num_object),
                cvt.Flatten()
            )

    def forward(self, ego_veh_state: elm.VehicleState,
                perception: Union[elm.TrackingBoxList, elm.OccupancyGrid2D],
                local_map: Union[elm.RoutedLocalMap, elm.LocalMap]) -> torch.Tensor:
        return self.encoder(ego_veh_state, perception, local_map)



if __name__ == '__main__':
    from spider.interface import DummyBenchmark
    obs = DummyBenchmark.get_environment_presets()
    encoder = KineStateEncoder()
    state = encoder(*obs)
    print(state)
    pass

