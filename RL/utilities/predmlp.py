import os
import pdb

# from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DCP_Agent.transition_model.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel_Pytorch

"""
输入是几台车的物理信息
输出是几台车的方向角转角和油门
流程是，运动信息输入，输出控制量，根据车辆模型更新运动信息，再输入
"""


class TrajPredMLP(nn.Module):
    """Predict one feature trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.LeakyReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.LeakyReLU(),
            # nn.Linear(hidden_unit, hidden_unit),
            # nn.LayerNorm(hidden_unit),
            # nn.LeakyReLU(),

            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x):
        # print("in mlp",x, self.mlp(x))
        return self.mlp(x)
    
    
class TrajPredGaussion(nn.Module):
    """Predict gaussion trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit, max_sigma=0.5, min_sigma=1e-4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ReLU(),
            # nn.Sigmoid(),
        )
        self.fc_mu = nn.Linear(hidden_unit, out_channels)
        self.fc_sigma = nn.Linear(hidden_unit, out_channels)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)

    def forward(self, x):
        x = self.fc(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        # sigma = self.fc_sigma(x)  
        # scaled of action should not be too large (i.e., max_sigma=10), it will never converge!
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    
class TrajPredGaussion_Intergrate(nn.Module):
    """Predict gaussion trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit, normalize_state_function, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.Sigmoid(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.Sigmoid(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.Sigmoid(),
            # nn.Linear(hidden_unit, hidden_unit)
        )
        self.fc_mu = nn.Linear(hidden_unit, out_channels)
        self.fc_sigma = nn.Linear(hidden_unit, out_channels)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
        
        # Vehicle Model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(80)
        self.dt = 0.1
        self.c_r = 0.01
        self.c_a = 0.05
        self.vehicle_model_torch = KinematicBicycleModel_Pytorch(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        
        self.normalize_state_function = normalize_state_function


    def forward(self, x):
        """
        qzl:
        transition model的本质是，输入s,a, 输出s'
        在这里的函数的意思是，
        输入x是观测量/状态s

        """
        
        normalize_x = self.normalize_state_function(x)
        normalize_x = self.fc(normalize_x)
        mu = self.fc_mu(normalize_x)
        sigma = torch.sigmoid(self.fc_sigma(normalize_x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        
        pred_state = self.forward_torch_vehicle_model(x, mu) # it is quite slow to use that
        
        return pred_state, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
        
    def forward_torch_vehicle_model(self, obs, pred_action):
        pred_state = []
        for i in range(len(pred_action[0])):         
            x = torch.mul(obs[i][0], self.obs_scale)
            y = torch.mul(obs[i][1], self.obs_scale)
            yaw = torch.mul(obs[i][4], self.obs_scale)
            v = torch.tensor(math.sqrt(torch.mul(obs[i][2], self.obs_scale) ** 2 + torch.mul(obs[i][3], self.obs_scale) ** 2))
            x, y, yaw, v, _, _ = self.vehicle_model_torch.kinematic_model(x, y, yaw, v, pred_action[0][i][0], pred_action[0][i][1])
            tensor_list = [torch.div(x, self.obs_scale), torch.div(y, self.obs_scale), torch.div(torch.mul(v, torch.cos(yaw)), self.obs_scale),
                           torch.div(torch.mul(v, torch.sin(yaw)), self.obs_scale), torch.div(yaw, self.obs_scale)]
            next_vehicle_state = torch.stack(tensor_list)
            # print("next_vehicle_state",next_vehicle_state)

            pred_state.append(next_vehicle_state)
            
        print("pred_state",pred_state)
        pred_state = torch.stack(pred_state)
        return pred_state
