from typing import Union, Sequence
from abc import abstractmethod
import torch
import torch.nn as nn

import spider.elements as elm
import spider.rl.convert as cvt



class BaseActionDecoder(nn.Module):
    @abstractmethod
    def forward(self, action_tensor:torch.Tensor,
                ego_veh_state:elm.VehicleState,
                perception:Union[elm.TrackingBoxList,elm.OccupancyGrid2D],
                local_map:Union[elm.RoutedLocalMap,elm.LocalMap]) -> elm.Trajectory:
        pass

class BaseActionEncoder(nn.Module):
    @abstractmethod
    def forward(self, traj:elm.Trajectory,
                ego_veh_state:elm.VehicleState,
                perception:Union[elm.TrackingBoxList,elm.OccupancyGrid2D],
                local_map:Union[elm.RoutedLocalMap,elm.LocalMap]) -> torch.Tensor:
        pass


class TrajActionDecoder(nn.Module):
    '''
    相对坐标下归一化的tensor，转化为绝对坐标下的elm.Trajectory
    '''
    def __init__(self, steps, dt, lon_range=(-80, 80), lat_range=(-80, 80), rotate=False):
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.action_dim = self.steps * 2

        self.linear_process = cvt.Compose(
            cvt.Reshape(steps, 2),
            cvt.DeNormalize(idx_range_dict={0: lon_range, 1: lat_range}),
            cvt.ZeroOffset(dim=-2),
        )
        self._rotate = rotate
        self.rotate_process = cvt.Rotate(0.0) if self._rotate else None


    def forward(self, traj_action:torch.Tensor, ego_veh_state:elm.VehicleState, *args) -> elm.Trajectory:
        '''
        forward的定义应该是decoder(action_tensor, *observation) -> trajectory
        '''
        traj_action = self.linear_process(traj_action) # 横纵向denormalize

        # if self.relative:
        #     raise NotImplementedError("Relative Trajectory Generation not supported for now, since it is temporally useless")
        # else:
        if self._rotate:
            self.rotate_process.set_angle(ego_veh_state.yaw())
            traj_action = self.rotate_process(traj_action)

        traj_array = traj_action.detach().cpu().numpy()
        traj_array[:,0] += ego_veh_state.x()
        traj_array[:,1] += ego_veh_state.y()

        traj = elm.Trajectory.from_trajectory_array(traj_array, self.dt, calc_derivative=True,
                          v0=ego_veh_state.v(), heading0=ego_veh_state.yaw(), a0=ego_veh_state.a())

        return traj


class TrajActionEncoder(nn.Module):
    '''
    绝对坐标下的elm.Trajectory转化为相对坐标下归一化的tensor
    '''
    def __init__(self, lon_range=(-80, 80), lat_range=(-80, 80), rotate=False): #, relative=False):
        super().__init__()
        # self.steps = steps
        # self.dt = dt
        # # self.relative = relative
        # self.action_dim = self.steps * 2
        #

        self._rotate = rotate
        self.rotate_process = cvt.Rotate(0.0) if self._rotate else None

        self.linear_process = cvt.Compose(
            cvt.Normalize(idx_range_dict={0: lon_range, 1: lat_range}),
        )


    def forward(self, trajectory: Union[elm.Trajectory, elm.FrenetTrajectory], *args, **kwargs) -> torch.Tensor:
        '''
        forward的定义应该是decoder(action_tensor, *observation) -> trajectory
        '''
        xs, ys = torch.FloatTensor(trajectory.x), torch.FloatTensor(trajectory.y)
        traj_action = (xs - xs[0], ys - ys[0])  # 平移， 初始点对齐0
        traj_action = torch.stack(traj_action, dim=-1)

        if self._rotate:
            self.rotate_process.set_angle(-trajectory.heading[0]) # 将轨迹旋转到与自车坐标下，即顺时针旋转yaw角
            traj_action = self.rotate_process(traj_action)

        traj_action = self.linear_process(traj_action) # 缩放到[0, 1]

        return traj_action.flatten() # 想想，要不要flatten()


class DiscreteTrajActionDecoder(nn.Module):
    def __init__(self, basic_candidates=(), sampler=None):
        super().__init__()
        # self.action_dim = sampler.num_samples + len(basic_candidates)
        self.sampler = sampler
        self.basic_candidates = basic_candidates
        self.action_dim = self.get_action_dim()
        assert self.action_dim > 0


    def get_action_dim(self):
        if self.sampler is None:
            return len(self.basic_candidates)
        else:
            return self.sampler.num_samples + len(self.basic_candidates)

    def _get_samples(self, *observation):
        if self.sampler is None:
            samples = []
        else:
            samples = self.sampler.sample_with_observation(*observation, frenet2cart=True, cart_order=2)
        return samples

    def forward(self, traj_action:torch.IntTensor, *observation) -> elm.Trajectory:
        idx = traj_action.item()

        if 0 <= idx < len(self.basic_candidates):
            return self.basic_candidates-idx
        elif idx < self.action_dim:
            samples = self._get_samples(*observation)
            return samples[idx]
        else:
            raise ValueError("Index out of bound.")


class DiscreteTrajActionEncoder(nn.Module):
    def __init__(self, basic_candidates=(), sampler=None):
        super().__init__()
        # self.action_dim = sampler.num_samples + len(basic_candidates)
        self.sampler = sampler
        self.basic_candidates = basic_candidates
        self.action_dim = self.get_action_dim()
        assert self.action_dim > 0

    def get_action_dim(self):
        if self.sampler is None:
            return len(self.basic_candidates)
        else:
            return self.sampler.num_samples + len(self.basic_candidates)

    def _get_samples(self, *observation):
        if self.sampler is None:
            samples = []
        else:
            samples = self.sampler.sample_with_observation(*observation, frenet2cart=True, cart_order=0)
        return samples

    def forward(self, pred_traj: elm.Trajectory, *observation):
        samples = self._get_samples(*observation)

        xys = torch.from_numpy(pred_traj.trajectory_array)
        all_xys = [torch.from_numpy(t.trajectory_array) for t in self.basic_candidates]
        all_xys += [torch.from_numpy(t.trajectory_array) for t in samples]
        all_xys = torch.stack(all_xys, dim=0)
        dists = torch.norm((all_xys - xys).flatten(-2, -1), dim=1)  # 欧式距离最小 torch.cdist(x, y)
        idx = torch.argmin(dists, dim=0)
        return idx


# class DiscreteTrajActionEncoder(nn.Module):
#     def __init__(self, discrete_num):
#         super().__init__()
#         self.action_dim = discrete_num
#
#     def forward(self, pred_traj: elm.Trajectory, traj_candidates):
#         xys = torch.from_numpy(pred_traj.trajectory_array)
#         all_xys = [torch.from_numpy(t.trajectory_array) for t in traj_candidates]
#         all_xys = torch.stack(all_xys, dim=0)
#         dists = torch.norm((all_xys - xys).flatten(-2, -1), dim=1)  # 欧式距离最小 torch.cdist(x, y)
#         idx = torch.argmin(dists, dim=0)
#         return idx

