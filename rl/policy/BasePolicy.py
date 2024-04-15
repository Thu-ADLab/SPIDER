import warnings
import time
import os

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from spider.data.data_factory import *

from abc import abstractmethod


try:
    import tensorboardX
except (ModuleNotFoundError, ImportError) as e:
    import spider
    tensorboardX = spider._virtual_import("tensorboardX", e)


# import subprocess
# import time
# import webbrowser
def _start_tensorboard(logdir='./tensorboard'):
    try:
        from tensorboard import program
    except (ModuleNotFoundError, ImportError) as e:
        warnings.warn("You have not correctly installed tensorboard. Please install tensorboard and open the UI manually")
        return

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, "--port", "6006"])  # 指定日志文件所在的目录
    url = tb.launch()

    # 打开浏览器并访问TensorBoard的可视化界面
    import webbrowser
    webbrowser.open(url, new=2)

    # import subprocess
    # process = subprocess.Popen(['tensorboard', '--logdir', logdir])
    # time.sleep(1)
    #
    # import webbrowser
    # webbrowser.open('http://localhost:6006', new=2)
    #
    # # 等待TensorBoard子进程结束
    # process.wait()

    # if input("Tensorboard has been launched. Type in 'q' or 'quit' to terminate the process.") in ['q','quit']:
    #     process.terminate()


def _get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _get_class_name(obj):
    return obj.__class__.__name__


class BasePolicy(nn.Module):
    def __init__(self, enable_tensorboard=False, tensorboard_root='./tensorboard/'):
        '''
        policy 本身应该是nn.Module，这可以视作是nn.Module的一种wrapper，
        用于在policy本身附加一个训练的流程（用于套用实现好的rl或il算法）
        '''
        super(BasePolicy, self).__init__()
        self.enable_tensorboard = enable_tensorboard
        self._tensorboard_root = tensorboard_root

        self._tensorboard_dir = os.path.join(self._tensorboard_root, _get_class_name(self)+_get_timestamp())
        self._writer = None #tensorboardX.SummaryWriter(self._tensorboard_dir)
        # else:
        #     self._tensorboard_dir = None
        #     self.writer = None

        self._activate_exp_buffer = False
        self._exp_buffer = None
        self._exp_extra_data = []


    @property
    def writer(self):
        if self._writer is None:
            self._writer = tensorboardX.SummaryWriter(self._tensorboard_dir)
        return self._writer

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except Exception:
            warnings.warn("You have set no param to the policy. Can not tell which device")
            return torch.device("cpu")

    @abstractmethod
    def forward(self, state):
        '''
        Must be implemented
        Forward the policy network to get an action for the state
        '''
        pass

    @abstractmethod
    def learn_batch(self, *one_batched_data) -> float:
        '''
        Must be implemented
        learn from one batch of data to update the parameters:
        calc loss, calc grad, back prog
        return the loss
        '''
        pass

    @abstractmethod
    def learn_dataset(self, epochs:int, train_loader:DataLoader, val_loader:DataLoader=None):
        '''
        Optionally implemented
        learn from the dataloader of all the dataset
        '''
        pass


    @abstractmethod
    def estimate_value(self, state, action=None):
        '''
        Optionally implemented for Critic/Value-based policy
        estimate value function given a state or a state-action pair
        '''
        pass

    def record_exp_extra_data(self, data):
        data = to_tensor(data)
        data = to_cpu(data)
        self._exp_extra_data.append(data)

    def start_tensorboard(self):
        if self._writer is not None:
            self._writer.flush() # 保证writer的内容全部写出，再打开tensorboard。

        import threading
        tensorboard_thread = threading.Thread(target=_start_tensorboard, args=(self._tensorboard_root,))
        tensorboard_thread.start()

    def reset_tensorboard_writer(self):
        if self._writer is not None:
            self._writer.close()
        self._tensorboard_dir = os.path.join(self._tensorboard_root, _get_class_name(self)+_get_timestamp())
        self._writer = None

    @abstractmethod
    def save_model(self, *args, **kwargs):
        '''
        Must be implemented
        Save the model
        '''
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        '''
        Must be implemented
        Load the model
        '''
        pass




class RLPolicy(BasePolicy):
    def __init__(self, enable_tensorboard=False, tensorboard_root='./tensorboard/'):
        super(RLPolicy, self).__init__(enable_tensorboard, tensorboard_root)

    def set_exploration(self, enable=True, *args, **kwargs):
        '''
        Optionally implemented
        Set the exploration strategy
        '''
        pass

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



