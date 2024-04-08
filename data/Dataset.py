import warnings
import tqdm
import os

import torch
from torch.utils.data import Dataset

import spider
from spider.data.data_factory import *


def get_record_num_for_segments(data_root):
    '''
    计算每个segment下的记录数量（也就是文件的数量）
    '''
    record_nums = {}
    for root, dirs, files in os.walk(data_root):
        for sub_dir in dirs:
            num = len(os.listdir(os.path.join(data_root, sub_dir)))
            # 注意, listdir也会把文件夹包含在内，这里是默认seg里面不包含文件夹
            record_nums[sub_dir] = num

    return record_nums


class OnlineDataset(Dataset):
    '''
    从在线的经验池创建Dataset
    '''
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class OfflineExpDataset(Dataset):
    '''
    从离线的数据集文件夹创建Dataset
    '''
    def __init__(self, data_root, file_format=spider.DATA_FORMAT_RAW, groundtruth=spider.DATA_GT_PLAN,
                 require_feedback=False, require_next_state=False):

        self.data_root = data_root
        self.file_format = file_format
        self.groundtruth_flag = groundtruth
        self.require_feedback = require_feedback
        self.require_next_state = require_next_state

        # self.record_nums = get_record_num_for_segments(data_root)

        # todo: 以下是建立index到文件名的映射。
        #  这里需要优化，因为一旦文件多的话扫描很慢，映射也会变慢, 而且占内存。可以考虑换成segment idx&record index
        self.index2file = {}
        count = 0
        for sub_dir_name in tqdm.tqdm(os.listdir(data_root), desc="Scanning all the sub directories"):
            sub_dir = os.path.join(self.data_root, sub_dir_name)
            if not os.path.isdir(sub_dir):
                print("Non-Directory is detected in the data_root. Ignoring {}".format(sub_dir))
                continue

            for filename in os.listdir(sub_dir):
                # 注意, listdir也会把文件夹包含在内，这里是默认seg里面不包含文件夹
                filepath = os.path.join(sub_dir, filename)
                if not os.path.isfile(filepath):
                    print("Non-File is detected in the data_root. Ignoring {}".format(filepath))
                    continue
                self.index2file[count] = filepath
                count += 1


    def __len__(self):
        return len(self.index2file)

    def __getitem__(self, index):
        filepath = self.index2file[index]
        raise NotImplementedError


class OfflineLogDataset(OfflineExpDataset):
    '''
    从离线的数据集文件夹创建Dataset
    '''
    def __init__(self, data_root, state_encoder, action_encoder,
                 file_format=spider.DATA_FORMAT_RAW, groundtruth=spider.DATA_GT_PLAN,
                 require_feedback=False, require_next_state=False):
        super(OfflineLogDataset, self).__init__(data_root, file_format, groundtruth, require_feedback, require_next_state)
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

    def __len__(self):
        return len(self.index2file)

    def __getitem__(self, index):
        filepath = self.index2file[index]
        log_record = self._load_data(filepath)
        state = self.state_encoder(*log_record[1])
        action = self._get_action(log_record)

        reward = done = torch.empty((0,))  # reward和done
        if self.require_feedback:
            reward = to_tensor(log_record[3]) if log_record[3] is not None else None
            done = to_tensor(log_record[4]) if log_record[4] is not None else None

        next_state = torch.empty((0,))  # next_state
        if self.require_next_state:
            if index + 1 > self.__len__() or log_record[4]: # 如果索引超界了，或者当前记录就是终止状态(done)
                next_state = torch.empty((0,))
            else:
                next_state = self._get_next_state(index)

        return state, action, reward, done, next_state

    def get_dataloader(self, *args, **kwargs):
        '''batch_size = 50,shuffle = False,sampler=None,batch_sampler = None,num_workers = 0,
        collate_fn = pin_memory = False,drop_last = False,timeout = 0,worker_init_fn = None,'''
        return torch.utils.data.DataLoader(self, *args, **kwargs)

    def get_record(self, index):
        filepath = self.index2file[index]
        log_record = self._load_data(filepath)
        return log_record


    def _load_data(self, pathfile):
        if self.file_format == spider.DATA_FORMAT_RAW:
            return load_raw(pathfile)
        elif self.file_format == spider.DATA_FORMAT_JSON:
            raise NotImplementedError("Not implemented. Please use RAW...")
        else:
            raise ValueError("Does not support this format: {}".format(self.file_format))

    def _get_action(self, log_record):
        if self.groundtruth_flag == spider.DATA_GT_PLAN:
            return self.action_encoder(log_record[2])
        elif self.groundtruth_flag == spider.DATA_GT_TRACE:
            raise NotImplementedError("Coming soon...") # todo: 下一步完成！

    def _get_next_state(self, current_index):
        filepath = self.index2file[current_index+1]
        next_log_record = self._load_data(filepath)
        next_state = self.state_encoder(next_log_record[1])
        return next_state


