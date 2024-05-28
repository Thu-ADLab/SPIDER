import torch
import torch.optim as optim
from copy import deepcopy
import random
import numpy as np
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt

import spider
from spider.rl.policy.BasePolicy import BasePolicy

class DiscretePPOPolicy(BasePolicy):
    def __init__(self, actor_network: torch.nn.Module, critic_network: torch.nn.Module, num_actions,
                 gamma=0.98, value_coef=0.5, entropy_coef=0.01,
                 lr_actor=3e-5, lr_critic=3e-5, max_grad_norm=1.0,
                 enable_tensorboard=False, tensorboard_root='./tensorboard/'):
        super().__init__(enable_tensorboard, tensorboard_root)

        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=lr_critic)

        self.num_actions = num_actions
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.clip_epsilon = 0.2

        self._plot_train_curve = False
        self._learning_count = 0

    def set_exploration(self, enable=True):
        # PPO does not have epsilon-greedy exploration, so this function is ignored
        pass

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        action_probs = self.actor_network(state.to(self.device))
        # Sample action from the distribution
        m = Categorical(action_probs)
        action = m.sample()

        if self._activate_exp_buffer:
            self.record_exp_extra_data(m.log_prob(action)) # add old log_prob to exp data

        # action = action_probs.argmax(dim=-1)
        return action

    def learn_batch(self, states, actions, rewards, dones, next_states, old_log_probs):
        self.train()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        ################### calc actor loss ###################
        # compute advantages：某个动作相对于当前状态的价值，减去状态的基准价值，即Q(s,a) - V(s)
        # 这里的advantage是用critic网络计算的，而不是用GAE计算的
        # advantage如果小于0，说明这个动作不好，应该减少这个动作的概率；反之，增加概率
        advantages = self._compute_advantages(states, rewards, dones, next_states)

        # value predictions：计算当前状态的价值
        values = self.critic_network(states)
        values = values.squeeze()

        # 执行的动作的log概率 log probabilities of actions
        action_probs = self.actor_network(states)
        m = Categorical(action_probs)
        log_probs = m.log_prob(actions).squeeze()

        # 计算重要性采样系数 IS = (pi_theta / pi_theta_old) = exp(log_pi_theta - log_pi_theta_old)
        ratios = torch.exp(log_probs - old_log_probs) # 这里错了

        # surrogate losses : IS * advantage
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()# 确保新策略（更新后）不会与旧策略偏离太远

        ################### calc critic loss ###################
        # 更新critic, 计算critic的loss (critic loss)
        value_targets = rewards + self.gamma * self.critic_network(next_states).squeeze() * (1 - dones)
        critic_loss = 0.5 * torch.mean((value_targets - values) ** 2) # MSE loss * 0.5

        # entropy loss
        entropy = m.entropy().mean()

        # Total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        ################### update parameters ###################
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()

        if getattr(self, "max_grad_norm", None) is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        ################### record/visualize ###################
        if self.enable_tensorboard:
            self.writer.add_scalar('loss/actor', actor_loss.item(), self._learning_count)
            self.writer.add_scalar('loss/critic', critic_loss.item(), self._learning_count)
            self.writer.add_scalar('loss/entropy', entropy.item(), self._learning_count)
            self.writer.add_scalar('loss/total', loss.item(), self._learning_count)

        if self._plot_train_curve:
            self._update_train_curve(loss.item())

        self._learning_count += 1

        return loss.item()


    def learn_buffer(self, exp_buffer:spider.data.ExperienceBuffer, batch_size, n_epochs=10):
        data_loader = exp_buffer.get_dataloader(batch_size, shuffle=True)
        for epoch in range(n_epochs):
            for batched_data in data_loader:
                self.learn_batch(*batched_data)


    def save_model(self, filename):
        torch.save({
            'actor_state_dict': self.actor_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict()
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_network.load_state_dict(checkpoint['critic_state_dict'])

    def _compute_advantages(self, states, rewards, dones, next_states):
        # A = Q(s,a) - V(s) = V(s') - V(s)
        # TD estimated V(s') = r + gamma * V(s')
        # Critic estimated V(s) = critic(s)
        with torch.no_grad():
            next_values = self.critic_network(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - self.critic_network(states).squeeze()
        return advantages#.cpu().numpy()


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

