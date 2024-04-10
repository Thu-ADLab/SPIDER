import torch
import torch.optim as optim

import tqdm
import matplotlib.pyplot as plt

import spider
from spider.rl.policy.BasePolicy import BasePolicy, DataLoader


class RegressionImitationPolicy(BasePolicy):
    def __init__(self, actor:torch.nn.Module, criterion:torch.nn.Module=None, lr=1e-4,
                 enable_tensorboard=False, tensorboard_root='./tensorboard/'):
        super().__init__(enable_tensorboard, tensorboard_root)
        self.actor = actor

        self.criterion = torch.nn.L1Loss() if criterion is None else criterion
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self._plot_train_curve = True

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        return self.actor(state.to(self.device))

    def learn_batch(self, batched_state, batched_target_action, *args):
        self.train()
        # forward
        batched_state = batched_state.to(self.device)
        batched_pred_action = self.forward(batched_state)

        # calculate loss
        batched_target_action = batched_target_action.to(self.device)
        loss = self.criterion(batched_pred_action, batched_target_action)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate_batch(self, batched_state, batched_target_action):
        self.eval()
        # forward
        batched_state = batched_state.to(self.device)
        batched_pred_action = self.forward(batched_state)

        # calculate loss
        batched_target_action = batched_target_action.to(self.device)
        loss = self.criterion(batched_pred_action, batched_target_action)
        return loss.item()

    def learn_dataset(self, epochs:int, train_loader:DataLoader, val_loader:DataLoader=None):
        '''
        Optionally implemented
        learn from the dataloader of all the dataset
        '''
        for epoch in tqdm.tqdm(range(epochs), desc="Training with dataset..."):
            avg_train_loss, count = 0.0, 0
            for batch_data in train_loader:
                avg_train_loss += self.learn_batch(*batch_data)
                count += 1
            avg_train_loss /= count
            if self.enable_tensorboard:
                self.writer.add_scalar('loss/train', avg_train_loss, epoch)

            if val_loader is not None:
                avg_val_loss, count = 0.0, 0
                for batch_data in val_loader:
                    avg_val_loss += self.learn_batch(*batch_data)
                    count += 1
                avg_val_loss /= count
                if self.enable_tensorboard:
                    self.writer.add_scalar('loss/val', avg_val_loss, epoch)
            else:
                avg_val_loss = None

            if self._plot_train_curve:
                self._update_train_curve(avg_train_loss, avg_val_loss)

        # if self.enable_tensorboard:
        #     self.start_tensorboard()
        plt.savefig('./train_curve.png')
        plt.close()


    def _update_train_curve(self, train_loss, val_loss=None):
        if not hasattr(self, "_loss_record"):
            self._loss_record = {"train":[], "val":[]}
        plt.cla()
        ax = plt.gca()

        self._loss_record["train"].append(train_loss)
        ax.plot(self._loss_record["train"],label="train")

        if val_loss is not None:
            self._loss_record["val"].append(val_loss)
        if len(self._loss_record["val"]) > 0:
            ax.plot(self._loss_record["val"], label="val")

        plt.legend()
        plt.pause(0.01)




