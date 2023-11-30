import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from typing import overload, Sequence, Union, Tuple

from spider.RL.dataset import ExperienceBuffer
from spider.RL.transition import KinematicObjectDeterministicTransition
from spider.elements.trajectory import Trajectory


class MBRLAgent:
    def __init__(self, state_veh_num, state_feat_num, action_dim, delta_t, reward_function,
                 buffer_size=100000, batch_size=32, gamma=0.98, lr=0.001, device=None):
        # RL相关
        self.state_size = state_veh_num * state_feat_num
        self.action_dim = action_dim

        # 数据集相关
        # qzl: 这里有个疑问，就是经验池应该算是planner里的，还是agent里面的
        self.experience = [None, None, None, None]  # 四元组,s,a,r,s'
        self.experience_buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size

        # 策略网络相关

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.gamma = gamma
        # self.q_network = QNetwork(state_size, action_size).to(self.device)
        # self.target_q_network = QNetwork(state_size, action_size).to(self.device)
        self.transition = KinematicObjectDeterministicTransition(
            state_veh_num, state_feat_num, action_dim, 64, delta_t
        ).to(self.device)
        self.reward_function = reward_function # 这个reward是单步控制量的reward

        self.optimizer = optim.Adam(self.transition.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        # self.steps = 0
        # self.target_update_frequency = target_update_frequency

    def set_transition(self, transition: nn.Module, lr=0.001):
        self.transition = transition.to(self.device)
        self.optimizer = optim.Adam(self.transition.parameters(), lr=lr)

    def save_transition_model(self, filename: str):
        torch.save(self.transition.state_dict(), filename)
        print("Successfully save the Q network model into ", filename)

    def load_transition_model(self, filename: str):
        state_dict = torch.load(filename)
        self.transition.load_state_dict(state_dict)  # 从本地加载模型
        print("Successfully load the Q network model from ", filename)

    def save_experience_buffer(self):
        pass

    def record_data(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        self.experience_buffer.store_experience(experience)


    def learn(self):
        batch_size = min([self.experience_buffer.size, self.batch_size])
        if batch_size <= 1:
            print("batch_size <= 1, Nothing to learn!")
            return

        # if self.experience_buffer.size < self.batch_size:
        #     return
        # batch_size = self.batch_size

        batch = self.experience_buffer.sample_batch(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        # rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        # dones = torch.FloatTensor(dones).to(self.device)

        # self.q_network.train()
        next_states_prediction = self.transition(states, actions)

        loss = self.criterion(next_states, next_states_prediction)
        loss.requires_grad_(True) # qzl:很奇怪 不知道为什么报错

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.steps += 1
        # if self.steps % self.target_update_frequency == 0:
        #     self.target_q_network.load_state_dict(self.q_network.state_dict())
        #     print("Target Q Network has updated at steps ", self.steps)

    def act(self, state: torch.Tensor, action_sequences: Union[torch.Tensor, Sequence[torch.Tensor]],
            egreedy:bool=False, epsilon:float=0.2) -> Tuple[int, torch.Tensor]:
        '''
        注意，action的定义是每一个时间步的控制量
        action_sequence是N个时间步的action序列
        action_sequences是M个这样的序列
        真正的action是对应action_sequence的第一步，类似MPC

        act函数返回的是action idx和 当前时间步的控制量
        '''
        # todo: 要不要加个random policy

        if egreedy and np.random.random() < epsilon:
            action_idx = np.random.randint(len(action_sequences))
            return action_idx, action_sequences[action_idx][0]

        Q_values = []

        for action_sequence in action_sequences:
            discount = 1.0
            Q = 0.0
            temp_state = state.clone().to(self.device)
            action_sequence = action_sequence.to(self.device)
            # todo: 把这个循环写进一个函数
            for action in action_sequence:
                # action = action.to(self.device)
                with torch.no_grad():
                    next_state = self.transition(temp_state, action)
                reward, done = self.reward_function.evaluate(temp_state, action, next_state)
                Q += reward * discount

                if done:
                    break
                temp_state = next_state # 进入下一状态
                discount *= self.gamma
            Q_values.append(Q)

        action_idx = np.argmax(Q_values).item()
        return action_idx, action_sequences[action_idx][0]



if __name__ == '__main__':
    from spider.RL.reward import FstateControlReward

    config = {
            "steps": 50,
            "dt": 0.1,
            "ego_veh_length": 5.0,
            "ego_veh_width": 2.0,
            "lane_width": 3.5,
            "finishing_line": 160,


            "state_veh_num": 10,
            "state_feat_num": 7,
            "discount_factor": 0.95,
            "closed_loop": True,


            "max_speed": 60/3.6,
            "min_speed": 0,
            "max_acceleration": 10,
            "max_deceleration": 10,
            # "max_centripetal_acceleration" : 100,
            "max_curvature": 100,
            "end_s_candidates": (15,40),
            "end_l_candidates": (-3.5,0,3.5), # s,d采样生成横向轨迹
            "end_v_candidates": tuple(i*60/3.6/2 + 1 for i in range(3)), # 改这一项的时候，要连着限速一起改了
            "end_T_candidates": (3,5) # s_dot, T采样生成纵向轨迹
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MBRLAgent(10,7,2,0.1, FstateControlReward(config))
    state = torch.rand(70).to(device)
    action_sequences = [torch.column_stack((torch.rand(20)*3,torch.rand(20)*20*3.14/180))  for _ in range(10)]

    action_idx, action = agent.act(state,action_sequences)
    print(action_idx)
    print(action)


