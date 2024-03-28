
import torch
import torch.nn as nn

import spider
from spider.data.DataBuffer import ExperienceBuffer

class MlpPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.device = torch.device("cpu")

        self.actor = nn.Sequential(
            nn.Linear(obs_space, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_space),
            nn.Tanh()
        )
        # self.action_head = nn.Linear(16, action_space * 2)

    def forward(self, x):
        return self.actor(x)


def test1():
    exp_buffer = ExperienceBuffer(
        file_prefix='exp',
        data_root="./dataset_exp/",
        autosave_max_intervals=50,
        file_format=spider.DATASET_FORMAT_TENSOR
        # file_format=spider.DATASET_FORMAT_RAW
    )

    policy = MlpPolicy(64, 64)

    exp_buffer.apply_to(policy)

    for step in range(168):
        obs = torch.rand(64)
        action = policy(obs)

    exp_buffer.release()



if __name__ == '__main__':
    test1()





