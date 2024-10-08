import random

import torch
import torch.optim as optim
from copy import deepcopy

import tqdm
import matplotlib.pyplot as plt

import spider
from spider.rl.policy.BasePolicy import BasePolicy

class DQNPolicy(BasePolicy):
    def __init__(self, q_network:torch.nn.Module, num_action,
                 target_update_frequency = 100, gamma=0.9, max_grad_norm=10, epsilon=0.1,
                 criterion:torch.nn.Module=None, lr=1e-4,
                 enable_tensorboard=False, tensorboard_root='./tensorboard/'):
        super().__init__(enable_tensorboard, tensorboard_root)

        self.num_action = num_action

        self.q_network = q_network #.to(self.device)
        self.target_q_network = deepcopy(self.q_network) #.to(self.device)

        self.target_update_frequency = target_update_frequency

        self.criterion = torch.nn.SmoothL1Loss() if criterion is None else criterion
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        # 优化器设置的是q_net的参数，切记。target net本身不做参数优化，只是通过load_state_dict来更新，用来估计下一状态的max Q

        self.max_grad_norm = max_grad_norm
        self.gamma = gamma

        self._egreedy = False
        self.epsilon = epsilon

        self._plot_train_curve = False
        self._learning_count = 0

    def set_exploration(self, enable=True, epsilon=0.2):
        self._egreedy = enable
        self.epsilon = epsilon

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        # the exp buffer will listen to it
        if self._egreedy and random.random()<self.epsilon:
            batch_size = state.shape[0]
            action = torch.randint(low=0, high=self.num_action, size=(batch_size,1))
        else:
            prob = self.q_network(state.to(self.device))
            action = torch.argmax(prob, dim=-1)
        return action


    def learn_batch(self, states, actions, rewards, dones, next_states):
        self.train()

        states, actions, rewards, dones, next_states = \
            (x.to(self.device) for x in (states, actions, rewards, dones, next_states))

        # Get current Q-values estimates
        q_values = self.q_network(states)  # Qs for current state
        # Retrieve the q-values for the actions from the replay buffer
        q_values = torch.gather(q_values, dim=1, index=actions.long())

        with torch.no_grad():
            # max(Qs for current state)
            q_values_next = self.target_q_network(next_states).max(dim=1)[0]
            # Avoid potential broadcast issue
            q_values_next = q_values_next.reshape(-1, 1)
            # 1-step TD
            q_targets = rewards + (1 - dones) * self.gamma * q_values_next

        loss = self.criterion(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        if getattr(self, "max_grad_norm", None) is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()


        if self.enable_tensorboard:
            self.writer.add_scalar('loss/Q_network', loss.item(), self._learning_count)

        if self._plot_train_curve:
            self._update_train_curve(loss.item())

        self._learning_count += 1
        if self._learning_count % self.target_update_frequency == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            print("Target Q Network has updated at steps ", self._learning_count)

        return loss.item()


    def save_model(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_q_network.load_state_dict(self.q_network.state_dict())


    def _update_train_curve(self, loss):
        if not hasattr(self, "_loss_record"):
            self._loss_record = {"train":[], "val":[]}
        plt.cla()
        ax = plt.gca()

        self._loss_record["train"].append(loss)
        ax.plot(self._loss_record["train"],label="train")
        plt.legend()
        plt.pause(0.01)


    def try_write_reward(self, reward, done=False, step=None):
        if self.enable_tensorboard:
            # record the reward of one step
            self.writer.add_scalar('reward/one_step', reward, step)

            # record the reward of one episode
            if not (hasattr(self, "_writer_acc_reward") and hasattr(self, "_writer_episode_count")):
                self._writer_acc_reward = 0.0
                self._writer_episode_count = 0
                self._writer_lifetime = 0
            self._writer_acc_reward += reward
            self._writer_episode_count += 1
            self._writer_lifetime += 1
            if done:
                self.writer.add_scalar('reward/episode', self._writer_acc_reward, self._writer_episode_count)
                self.writer.add_scalar('reward/lifetime', self._writer_lifetime, self._writer_episode_count)
                self._writer_acc_reward = 0.0
                self._writer_lifetime = 0


