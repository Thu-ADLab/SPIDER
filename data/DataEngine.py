import os
from collections import deque
import torch
import torch.nn as nn

import spider
from spider.data.common import *




class DataEngine:
    def __init__(self,
                 buffer_size=1000000,
                 save_data=True,  # 只有save data为True时，以下参数才有意义
                 episode_save=True,
                 data_root='./dataset/',
                 file_format=spider.DATASET_FORMAT_JSON,
                 file_prefix='log',
                 max_log_saves=1000000,
                 data_augumentor=None, # 数据增广
                 data_labeller=None # 自动标注
                 ):

        # online log buffer
        self.log_buffer = deque(maxlen=buffer_size)



        # data factory
        self.data_factory = None

        # data augumentor
        self.data_augumentor = data_augumentor

        # offline datasets settings
        self.save_data: bool = save_data
        self.episode_save:bool = episode_save # done信号了要不要保存一次数据到本地
        self.data_root = data_root
        self.filename_prefix = file_prefix
        self.file_format = file_format
        self.log_index = 0
        self.max_log_saves = int(max_log_saves)
        self._filename_index_length = len(str(self.max_log_saves-1)) # 判断几位数的方式好像不太好

        if self.save_data:
            self._prepare_data_root()




    def log(self, timestamp, ego_state, perception, local_map, action=None, reward=0.0, done=False):
        '''
        log the origin observation and action into the online buffer
        todo:以后要考虑压缩local_map, 尤其是导航信息太多了。
        '''
        self.log_buffer.append((timestamp, ego_state, perception, local_map, action, reward, done))
        enough_data_signal = False

        if self.save_data: # qzl这一段是不是不应该放到log函数里面？应该放在collect_data?再想想吧
            if self._need_to_save(len(self.log_buffer), done):
                self.save_log()
                self.clear_buffer()
                # todo:不该清空？应该把计数器清零，但deque内保存的内容不清空，否则在线经验池不就没了吗？

                self.log_index += 1
                if self.log_index>= self.max_log_saves:
                    enough_data_signal = True
                    print("DEBUG: Already saved {} log records successfully!".format(self.log_index))
        return enough_data_signal


    def save_log(self, log_index=None):
        '''
        save the log buffer to the offline dataset
        '''
        log_index = self.log_index if log_index is None else log_index

        pass

    def augument_data(self):
        pass

    def collect_data(self, spider_planner, environment_interface, num_steps, augumentation=False):
        '''
        用planner在环境中收集数据，并保存至本地数据集，
        直到达到num_steps条数据（目前仅支持按step计数，未来要加入以log数量计数）
        :param spider_planner:
        :param environment_interface:
        :param num_steps:
        :return:
        '''
        pass

    def replay_log(self):
        pass

    def visualize_log(self):
        pass

    @staticmethod
    def load_from_datasets(data_root, file_format):
        pass


    def load_from_buffer(self, data_root, file_format):
        pass


    def clear_buffer(self):
        self.log_buffer.clear()

    def shuffle_buffer(self):
        pass

    def _need_to_save(self, buffer_length, done):
        if buffer_length >= self.log_buffer.maxlen:
            return True
        elif self.episode_save and done:
            return True
        else:
            return False


    def _get_filepath(self, log_index):
        filename = self.filename_prefix + str(log_index).zfill(self._filename_index_length)\
                   + format_suffix[self.file_format]
        return os.path.join(self.data_root, filename)

    def _prepare_data_root(self):
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        elif len(os.listdir(self.data_root)) == 0:
            response = input("WARNING: the target data root directory {} is not empty! "
                             "Type in \"yes\" to overwrite everything and continue")
            assert response == "yes", "please empty the target data root directory or " \
                                      "change the data_root setting."

