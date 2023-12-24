import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

import torch.optim as optim
import torch.nn.functional as F

import spider
from spider.RL.dataset import ExperienceBuffer
from spider.RL.agents.BaseAgent import BaseAgent

'''
qzl
'''

class QNetwork(nn.Module):
    '''
    一个简单的 MLP Qnetwork
    '''
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, buffer_size=100000, batch_size=32, gamma=0.98, lr=0.001,
                 target_update_frequency=50, device=None):
        super(DQNAgent, self).__init__(buffer_size, batch_size, gamma, lr, device)

        # RL相关
        self.state_size = state_size
        self.action_size = action_size

        # 数据集相关
        # qzl: 这里有个疑问，就是经验池应该算是planner里的，还是agent里面的
        self.experience = [None, None, None, None, None]  # 5元组,s,a,r,s',d
        self.experience_buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size

        # 策略网络相关

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.gamma = gamma
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_q_network = deepcopy(self.q_network).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr) # 直接更新的参数是q_network，target网络一段时间后更新
        self.criterion = nn.MSELoss()
        self.steps = 0
        self.target_update_frequency = target_update_frequency

        self.mode_flag = spider.NN_TRAIN_MODE

    def train_mode(self):
        self.q_network.train()
        self.target_q_network.train()
        self.mode_flag = spider.NN_TRAIN_MODE

    def eval_mode(self):
        self.q_network.eval()
        self.target_q_network.eval()
        self.mode_flag = spider.NN_EVAL_MODE


    def set_network(self, q_network: nn.Module, lr=0.001):
        self.q_network = q_network.to(self.device)
        self.target_q_network = q_network.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def save_model(self, filename:str):
        torch.save(self.q_network.state_dict(), filename)
        print("Successfully save the Q network model into ", filename)

    def load_model(self, filename:str):
        state_dict = torch.load(filename)
        self.q_network.load_state_dict(state_dict)   # 从本地加载模型
        self.target_q_network.load_state_dict(state_dict)
        print("Successfully load the Q network model from ", filename)

    def learn(self):
        batch_size = min([self.experience_buffer.size, self.batch_size])
        if batch_size <= 1:
            print("batch_size <= 1, Nothing to learn!")
            return

        if self.mode_flag != spider.NN_TRAIN_MODE:
            self.train_mode()

        # if self.experience_buffer.size < self.batch_size:
        #     return
        # batch_size = self.batch_size

        batch = self.experience_buffer.sample_batch(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # self.q_network.train()
        q_values = self.q_network(states)
        q_values_next = self.target_q_network(next_states).max(dim=1)[0]

        q_targets = rewards + (1 - dones) * self.gamma * q_values_next

        q_values_for_actions = q_values.gather(dim=1, index=actions.unsqueeze(1))

        loss = self.criterion(q_values_for_actions, q_targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            print("Target Q Network has updated at steps ", self.steps)


    def act(self, state, egreedy=False, epsilon=0.1):
        if self.mode_flag != spider.NN_EVAL_MODE:
            self.eval_mode()

        if egreedy and np.random.random() < epsilon:
            action_idx = np.random.randint(self.action_size)
        else:
            state = state.to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            action_idx = torch.argmax(q_values).item()

        if not egreedy:
            print(action_idx, q_values[action_idx].item())
        return action_idx


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(100, 10,)
    inp = torch.rand(100).to(device)
    action_idx = agent.act(inp)
    print(action_idx)
    out = agent.q_network(inp).cpu()
    print(out)


