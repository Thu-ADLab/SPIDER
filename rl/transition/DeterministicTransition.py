import torch
import torch.nn as nn
import math

from spider.control.vehicle_model import Bicycle

# class ObjectDeterministicTransition(nn.Module):
#     '''
#     一个简单的 MLP network
#     '''
#     def __init__(self, state_veh_num, state_feat_num, action_dim, hidden_dim):
#         super(ObjectDeterministicTransition, self).__init__()
#         state_dim = state_feat_num * state_veh_num
#         self.MLP = nn.Sequential(
#             nn.Linear(state_dim + action_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, state_dim),
#         )
#
#
#     def forward(self, state, action):
#         ego_control = torch.tensor(action)
#         x = torch.cat((state, ego_control))
#         delta = self.MLP(x)
#         next_state = delta + state
#         # next_state = self.MLP(state, action) # qzl:这个似乎不太好
#         return next_state


class KineTransition(nn.Module):
    '''
    认为每个物体都是车
    '''

    def __init__(self, state_veh_num, state_feat_num, action_dim, hidden_dim, delta_t,
                 max_accleration=5.0, max_steer=35*math.pi/180):
        super(KineTransition, self).__init__()
        self.state_veh_num = state_veh_num
        self.state_feat_num = state_feat_num

        state_dim = state_feat_num * state_veh_num
        self.MLP = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, (state_veh_num-1) * 2),
            nn.Tanh() # 缩放到-1和1
        )

        self.dt = delta_t

        self.max_accleration = max_accleration # todo:现在暂时写的是正负约束相同
        self.max_steer = max_steer

    def forward(self, state, action):
        '''
        action是加速度和转向角。
        注意，action不用归一化
        '''
        controls = self._forward_control(state, action) # 输出的是车的控制
        next_state = self._conduct_controls(state, controls)
        return next_state

    def _forward_control(self, state, action):
        # qzl: 需要考虑一下action会不会因为归一化而产生缩放的问题
        ego_control = action if type(action) is torch.Tensor else torch.tensor(action).to(state.device)

        x = torch.cat((state, ego_control), dim=-1)
        results = self.MLP(x)

        batch_size = state.shape[0] if len(state.shape)>1 else 1

        controls = results.view(batch_size, self.state_veh_num-1, 2)
        controls[:, :, 0] *= self.max_accleration
        controls[:, :, 1] *= self.max_steer

        controls = torch.cat((ego_control.view(batch_size, 1, 2), controls), dim=1)
        return controls

    def _conduct_controls(self, state, controls):
        batch_size = state.shape[0] if len(state.shape) > 1 else 1

        next_state = state.view(batch_size, self.state_veh_num, self.state_feat_num)

        for batch in range(len(controls)):
            batch_controls = controls[batch]
            for i, control in enumerate(batch_controls):
                acc, steer = control
                presence, x, y, length, width, heading, speed = next_state[batch, i, :]#.detach()
                if not presence:
                    continue
                veh = Bicycle(x, y, speed, 0.0, heading, dt=self.dt)
                veh.step(acc, steer)
                next_state[batch, i, :] = torch.tensor([presence, veh.x, veh.y, length, width, veh.heading, veh.velocity])

        if len(state.shape) > 1:
            next_state = next_state.view(batch_size, -1)
        else:
            next_state = next_state.flatten()
        return next_state


if __name__ == '__main__':
    trans = KineTransition(10, 7, 2, 64, 0.1)
    state = torch.rand(70)
    action = [1.0, 0.0]
    next_state = trans(state, action)
    print(next_state)
    print(next_state.shape)
    pass
