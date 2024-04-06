import math
import warnings
from typing import Union, List, Tuple

import torch
import torch.nn as nn

import spider.elements as elm
from spider.utils.transform.relative import RelativeCoordinateTransformer

class DataType: # no need to use Enum, for the value can be calculated
    Any = -1 # -1 & x = x, -1 | x =-1
    Undefined = 0
    Observation = 1
    State = 2
    Action = 4
    Tensor = State | Action
    Plan = 8


class Compose(nn.Module):
    _input_type = DataType.Any
    _output_type = DataType.Any

    def __init__(self, *converters: nn.Module):
        super().__init__()
        self.converters = list(converters)
        # self.composed_converter = nn.Sequential(*converters)

        in_types = [getattr(converter, "_input_type", DataType.Any) for converter in converters]
        out_types = [getattr(converter, "_output_type", DataType.Any) for converter in converters]
        for i, (in_type, out_type) in enumerate(zip(in_types, out_types)):
            if i == 0:
                continue
            if not (in_type & out_types[i-1]):
                warnings.warn("Mismatching input/output type in Compose: {}, {}".format(out_types[i-1], in_type))
        self._input_type = in_types[0]
        self._output_type = out_types[-1]


    def forward(self, *args):
        # return self.composed_converter(*args)
        x = args
        for i, cvt in enumerate(self.converters):
            if isinstance(x, tuple):
                x = cvt(*x)
            else:
                x = cvt(x)
        return x

    def append(self, converter):
        self.converters.append(converter)


class ToKineState(nn.Module):
    _input_type = DataType.Observation
    _output_type = DataType.State

    def __init__(self, max_num_object=10, max_num_waypoint=50, relative=True):
        super(ToKineState, self).__init__()
        self.max_num_object = max_num_object
        self.max_num_waypoint = max_num_object
        self.relative = relative
        self.tf = RelativeCoordinateTransformer() if relative else None

    def forward(self,
                ego_veh_state: elm.VehicleState,
                perception: Union[elm.TrackingBoxList, elm.OccupancyGrid2D],
                local_map: Union[elm.RoutedLocalMap, elm.LocalMap],
                ) -> torch.Tensor:
        """
            # veh_info: [[presence, x, y, length, width, yaw, vx, vy] * Nv]
            # refer_line: [[x, y] * Nw]
        """

        ego_info = [
            1.0, *ego_veh_state.obb, ego_veh_state.velocity.x, ego_veh_state.velocity.y
        ]
        self.tf.set_ego_pose(ego_veh_state.x(), ego_veh_state.y(), ego_veh_state.yaw())
        self.tf.set_ego_velocity(ego_veh_state.velocity.x, ego_veh_state.velocity.y)

        assert isinstance(perception, elm.TrackingBoxList), "Only TrackingBoxList is supported for object state"
        all_obj_info = [ego_info]
        num_valid_veh = min([len(perception), self.max_num_object - 1])
        for i in range(num_valid_veh):
            tbox = perception[i]
            if self.relative:
                x,y,yaw,vx,vy = self.tf.abs2rel(tbox.x, tbox.y, tbox.box_heading, tbox.vx, tbox.vy)
                all_obj_info.append([1.0, x, y, tbox.length, tbox.width, yaw, vx, vy])
            else:
                all_obj_info.append([1.0, *tbox.obb, tbox.vx, tbox.vy])

        states = torch.tensor(all_obj_info, dtype=torch.float32)
        return states




# class ToWaypointsAction(nn.Module):
#     _input_type = DataType.Plan
#     _output_type = DataType.Action
#     def __init__(self, max_num_waypoint=50, relative=True):
#         super().__init__()


# class ToRelative(nn.Module):
#     _input_type = DataType.State | DataType.Action
#     _output_type = DataType.State | DataType.Action
#     def __init__(self):
#         super().__init__()
#         self.tf = RelativeCoordinateTransformer()
#
#     def forward(self, tensor:torch.Tensor) -> torch.Tensor:
#         return tensor

class Normalize(nn.Module):
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, global_range:Union[List, Tuple]=None, idx_range_dict:dict={}, dim=-1, clamp=False):
        '''
        global_range: [min, max] to normalize the entire tensor
        idx_range_dict: dict of {idx: [min, max]}
        dim: the dimension to be normalized ( to be indexed ). By default, it is the last dimension
        clamp: whether to clamp the output into the range [0,1]
        '''
        super(Normalize, self).__init__()
        self.global_range = global_range # 全部数据首先都按该范围缩放
        self.idx_range_dict = idx_range_dict
        self.dim = dim
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.clone()

        # 数据全局范围缩放
        if self.global_range is not None:
            min_val, max_val = self.global_range
            if self.clamp: # Clip to the specified range
                norm_x.clamp_(min_val, max_val) # .clamp_ 是直接修改当前张量的值
            norm_x -= min_val
            norm_x /= max_val - min_val

        # 将对应索引的切片，分别缩放到不同的范围
        for idx, (min_val, max_val) in self.idx_range_dict.items():
            target_slice = x.select(self.dim, idx) # 注意，select返回的是引用赋值的索引
            if self.clamp:
                target_slice = target_slice.clamp(min_val, max_val) # .clamp 是直接返回一个新的张量
            norm_slice = target_slice - min_val
            norm_slice /= max_val - min_val
            norm_x.select(self.dim, idx).copy_(norm_slice) # 赋值

        return norm_x


class DeNormalize(nn.Module):  # The inverse of normalize
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, global_range:Union[List, Tuple]=None, idx_range_dict:dict={}, dim=-1, clamp=False):
        '''
        global_range: [min, max] to De-normalize the entire tensor
        idx_range_dict: dict of {idx: [min, max]}
        dim: the dimension to be de-normalized ( to be indexed ). By default, it is the last dimension
        clamp: whether to clamp the output into the given range
        '''
        super(DeNormalize, self).__init__()
        self.global_range = global_range # 全部数据首先都按该范围缩放
        self.idx_range_dict = idx_range_dict
        self.dim = dim
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        de_norm_x = x.clone()

        # 数据全局范围缩放
        if self.global_range is not None:
            if self.clamp:
                de_norm_x.clamp_(0.0, 1.0)
            min_val, max_val = self.global_range
            de_norm_x *= max_val - min_val
            de_norm_x += min_val

        # 将对应索引的切片，分别缩放到不同的范围
        for idx, (min_val, max_val) in self.idx_range_dict.items():
            target_slice = x.select(self.dim, idx) # 注意，select返回的是引用赋值的索引
            if self.clamp:
                target_slice = target_slice.clamp(0.0, 1.0) # .clamp 是直接返回一个新的张量
            de_norm_slice = target_slice * (max_val - min_val)
            de_norm_slice += min_val
            de_norm_x.select(self.dim, idx).copy_(de_norm_slice) # 赋值
        return de_norm_x


class Rotate(nn.Module):
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, angle):
        # 旋转二维向量（或二维的矩阵的每一行）指定的角度
        super(Rotate, self).__init__()
        self.angle = angle
        self.transposed_rot_mat = torch.tensor([[math.cos(angle), math.sin(angle)],
                                                [-math.sin(angle), math.cos(angle)]])

    def set_angle(self, angle):
        self.angle = angle
        self.transposed_rot_mat = torch.tensor([[math.cos(angle), math.sin(angle)],
                                                [-math.sin(angle), math.cos(angle)]])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[-1] == 2, "The input tensor should be 2D vector or 2D matrix"
        rotated_tensor = torch.matmul(tensor, self.transposed_rot_mat.to(tensor.device))
        return rotated_tensor


class ZeroOffset(nn.Module):
    '''
    将某个（序列）张量，在某个维度上，选定某个索引，其他维度减去该索引对应的值，
    即数据整体加减，使得某一维度的某一索引切片的值为0
    '''
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, dim, target_idx=0): # 默认序列第一个为0
        super(ZeroOffset, self).__init__()
        self.dim = dim
        self.target_idx = torch.tensor(target_idx) # tensor() because index_select() says so...

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        target_zero_slice = tensor.index_select(self.dim, self.target_idx.to(tensor.device))  # 获取目标维度的切片
        tensor = tensor - target_zero_slice
        return tensor


class Shift(nn.Module):  # 张量平移
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, shift_vector, dim=-1):
        """
        Args:
            shift_vector: 一个张量，表示平移的大小。如果维度为1，则会应用到所有的元素。
            dim: 要进行平移的维度。默认为最后一个维度。
        """
        super(Shift, self).__init__()
        self.shift_vector = shift_vector
        self.dim = dim

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x: 输入的张量，可以是任意形状的张量。

        Returns:
            shifted_x: 平移后的张量。
        """
        # 确保 shift_vector 和输入张量 x 的维度匹配
        if self.shift_vector.dim() == 1 and x.dim() > 1:
            # 在 shift_vector 上增加 batch 维度
            batch_size = x.size(0) if x.dim() > 1 else 1
            self.shift_vector = self.shift_vector.unsqueeze(0).expand(batch_size, -1)

        # 使用 torch.roll 进行平移操作
        shifted_x = torch.roll(x, shifts=self.shift_vector, dims=self.dim)

        return shifted_x


class Pad(nn.Module):
    '''
    pad a tensor with additional channel num at a specific dimension
    '''
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, dim, padding_channels=1, padding_value=0.0):
        super(Pad, self).__init__()
        self.dim = dim
        self.padding_channels = padding_channels
        self.padding_value = padding_value

    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        # pad 的 shape
        pad_shape = list(tensor.shape)
        pad_shape[self.dim] = self.padding_channels

        padding_tensor = torch.full(pad_shape, self.padding_value, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, padding_tensor], dim=self.dim)

        return padded_tensor


class Truncate(nn.Module):
    '''
    Truncate a tensor to a fixed length if it is too long
    '''
    _input_type = DataType.State | DataType.Action
    _output_type = DataType.State | DataType.Action

    def __init__(self, dim, target_length):
        super().__init__()
        self.dim = dim
        self.target_length = target_length

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[self.dim] > self.target_length:
            tensor = tensor.narrow(self.dim, 0, self.target_length)
        return tensor


class FixLength(nn.Module):
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, dim, target_length, padding_value=0.0):
        super().__init__()
        self.dim = dim
        self.target_length = target_length
        self._pad = Pad(dim, padding_value=padding_value)
        self._truncate = Truncate(dim, target_length)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[self.dim] > self.target_length:
            tensor = self._truncate(tensor)
        elif tensor.shape[self.dim] < self.target_length:
            self._pad.padding_channels = self.target_length - tensor.shape[self.dim]
            tensor = self._pad(tensor)
        return tensor


class Flatten(nn.Module):
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor.flatten()


class Reshape(nn.Module):
    _input_type = DataType.Tensor
    _output_type = DataType.Tensor
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor.reshape(self.shape)


class ToRelativeObservation(nn.Module):
    '''
    qzl: I suggest not using it!
    '''
    _input_type = DataType.Observation
    _output_type = DataType.Observation

    def __init__(self, max_num_object=math.inf):
        super().__init__()
        self.tf = RelativeCoordinateTransformer()
        self.max_num_object = max_num_object

    def forward(self,
                ego_veh_state: elm.VehicleState,
                perception: Union[elm.TrackingBoxList, elm.OccupancyGrid2D],
                local_map: Union[elm.RoutedLocalMap, elm.LocalMap]) -> elm.Observation:
        ego_pose = [ego_veh_state.x(), ego_veh_state.y(), ego_veh_state.yaw()]
        ego_vel = [ego_veh_state.velocity.x, ego_veh_state.velocity.y]
        self.tf.set_ego_pose(*ego_pose)
        self.tf.set_ego_velocity(*ego_vel)

        rel_perception = None
        if isinstance(perception, elm.TrackingBoxList):
            rel_perception = elm.TrackingBoxList()
            for i, tbox in enumerate(rel_perception):
                tbox:elm.TrackingBox
                x,y,yaw,vx,vy = self.tf.abs2rel(tbox.x, tbox.y, tbox.box_heading, tbox.vx, tbox.vy)
                rel_perception.append( # 转换完了之后prediction和history信息会丢失，因为是重新生成
                    elm.TrackingBox([x,y,tbox.length,tbox.width,yaw], vx, vy, tbox.id, tbox.class_label))
        else:
            raise NotImplementedError("Currently only support TrackingBoxList")

        rel_local_map = local_map.copy() # todo:补充local map的相对坐标转换

        return ego_veh_state, rel_perception, rel_local_map







if __name__ == '__main__':
    tensor = torch.tensor([
        [1,2,3,4],
        [2,4,6,8],
        [3,6,9,10.]
    ])

    normalize = Normalize([1, 9], idx_range_dict={1: [3, 5], 3: [2, 8]}, dim=-1, clamp=True)

    output = normalize(tensor)

    print("原始 tensor:")
    print(tensor)
    print("\n处理后的 tensor:")
    print(output)

    # import torchvision.transforms as transforms
    #
    # train_transformer = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(256),
    #     # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])


    pass

