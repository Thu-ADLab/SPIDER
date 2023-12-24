import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from abc import abstractmethod

import spider
from spider.RL.dataset import ExperienceBuffer
import torch.optim as optim
import torch.nn.functional as F


class BaseAgent:
    def __init__(self, buffer_size=100000, batch_size=32, gamma=0.98, lr=0.001, device=None):
        # 数据集相关
        # qzl: 这里有个疑问，就是经验池应该算是planner里的，还是agent里面的
        self.experience = [None, None, None, None, None]  # 5元组,s,a,r,s',d
        self.experience_buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.gamma = gamma

        self.optimizer = None
        self.criterion = None

        self.mode_flag = spider.NN_TRAIN_MODE
        pass

    def record_data(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        self.experience_buffer.store_experience(experience)

    @abstractmethod
    def train_mode(self):
        '''
        把自己的网络放进来
        '''
        self.mode_flag = spider.NN_TRAIN_MODE

    @abstractmethod
    def eval_mode(self):
        '''
        把自己的网络放进来
        '''
        self.mode_flag = spider.NN_EVAL_MODE

    @abstractmethod
    def set_network(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_experience_buffer(self):
        pass




    def learn(self, *args, **kwargs):
        pass


    def act(self, *args, **kwargs):
        pass

