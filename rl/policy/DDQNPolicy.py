import torch


from spider.rl.policy.DQNPolicy import DQNPolicy

class DDQNPolicy(DQNPolicy):
    def learn_batch(self, states, actions, rewards, dones, next_states):
        self.train()

        states, actions, rewards, dones, next_states = \
            (x.to(self.device) for x in (states, actions, rewards, dones, next_states))

        # Get current Q-values estimates
        q_values = self.q_network(states)  # Qs for current state
        # Retrieve the q-values for the actions from the replay buffer
        q_values = torch.gather(q_values, dim=1, index=actions.long())

        # DDQN: use the Q-network to select the action, and use the target Q-network to evaluate the action
        next_action = self.q_network(next_states).argmax(dim=1, keepdim=True)

        with torch.no_grad():
            # max(Qs for current state)
            q_values_next = self.target_q_network(next_states).gather(1, next_action)
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

