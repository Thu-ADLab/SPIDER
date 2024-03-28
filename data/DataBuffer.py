import itertools
import warnings
from typing import Deque, Tuple
import random
from collections import deque
from abc import abstractmethod
import os
import json
import pickle

import torch
import torch.nn as nn


import spider
import spider.elements as elm
from spider.data.common import format_suffix
from spider.data.decorators import *

_default_max_len = 10000
_default_max_saves = 10000

# todo：以后加一个更加通用的databuffer，以及其装饰器。存任意形式的数据，并且保存到本地数据集。支持任意形式的函数的输入



class BaseBuffer(deque):
    def __init__(self,
                 maxlen=_default_max_len,
                 # 保存到本地数据集的参数
                 data_root='./dataset/',
                 file_prefix='data',
                 file_format=spider.DATASET_FORMAT_RAW,
                 # 下面是自动保存离线数据集时候的参数
                 autosave=True,  # 只有autosave为True时，以下参数才有意义
                 autosave_max_intervals=1000,
                 autosave_episode_done=True,  # done信号了要不要保存一次数据到本地, 如果不保存就是固定intervals
                 max_saves=_default_max_saves,
                 ):

        super(BaseBuffer, self).__init__(maxlen=maxlen)

        self._autosave: bool = autosave
        self._autosave_episode_done: bool = autosave_episode_done  # done信号了要不要保存一次数据到本地
        self._autosave_max_intervals: int = autosave_max_intervals
        assert autosave_max_intervals <= maxlen, "autosave_max_intervals should not be larger than maxlen"

        self.data_root = data_root
        self.filename_prefix = file_prefix
        self.file_format = file_format

        self._count_saves = 0
        self.max_saves = int(max_saves)

        self._filename_index_length = len(str(self.max_saves - 1))  # 判断几位数的方式好像不太好

        self._save_target_slice = [0, 0] # 记录当前episode的起始和结束位置(左闭右开)

        self._temp_record = { # 暂时储存一条数据，存满了就放入buffer
            # "timestamp": None,
            "forward": None,
            "feedback": None
        }
        self._STORE_FORWARD_ONLY = False
        # self._STORE_FORWARD_FEEDBACK = False
        # _STORE_FORWARD_ONLY若为True，则record_forward后自动存入buffer，清除record
        # _STORE_FORWARD_ONLY若为False，则record完毕forward后，等待record_feedback后再自动存入buffer，清除record

        if self._autosave:
            self._prepare_data_root()

    # def record_timestamp(self, timestamp):
    #     self._temp_record["timestamp"] = float(timestamp)


    def record_forward(self, *args, **kwargs):
        '''timestamp, obs, plan'''
        self._temp_record["forward"] = [args, kwargs]

        if self._STORE_FORWARD_ONLY:
            # _STORE_FORWARD_ONLY若为True，则record_forward后自动存入buffer，清除record
            self.store(*args, **kwargs)
            self._clear_record()
        else:
            pass  # wait for record_feedback...


    def record_feedback(self, *args, **kwargs):
        '''reward, done'''
        if self._temp_record["forward"] is None:
            return  # 只有在有forward信息时才存feedback信息

        self._temp_record["feedback"] = [args, kwargs]

        if not self._STORE_FORWARD_ONLY:
            # _STORE_FORWARD_ONLY若为True，则record_forward后自动存入buffer，清除record
            fw_args, fw_kwargs = self._temp_record["forward"]
            self.store(*fw_args, *args, **fw_kwargs, **kwargs)
            self._clear_record()

    @abstractmethod
    def store(self, *args, **kwargs):
        pass

    # def store_record(self):
    #     ''' store the temp record'''
    #     # for val in self._temp_record.values():
    #     #     if val is None:
    #     #         self._clear_record()
    #     #         continue
    #
    #     self.store(*self._temp_record.values())
    #     self._clear_record()

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def replay(self, *args, **kwargs): # todo:想清楚replay函数到底是想实现什么功能
        pass

    @abstractmethod
    def apply_to(self, *args, **kwargs):
        pass # 用decorator监听模块

    @abstractmethod
    def sample(self,*args,**kwargs):
        pass

    def clear(self) -> None:
        super(BaseBuffer, self).clear()
        self._save_target_slice = [0, 0]

    def release(self):
        if self._autosave and not self.enough_saves():
            if self._save_target_slice[1] - self._save_target_slice[0]>0:
                self.save()
        self.clear()


    def shuffle(self):
        random.shuffle(self)
        self._save_target_slice = [len(self), len(self)]

    def get_list_by_slice(self, start, end):
        return list(itertools.islice(self, start, end))

    def enough_saves(self) -> bool:
        # 检查是否已经保存了足够的数据，如果达到阈值，则不再保存
        return self._count_saves >= self.max_saves




    def _get_filepath(self, idx):
        filename = self.filename_prefix + str(idx).zfill(self._filename_index_length)\
                   + format_suffix[self.file_format]
        return os.path.join(self.data_root, filename)

    def _prepare_data_root(self):
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        elif len(os.listdir(self.data_root)) > 0:
            response = input("WARNING: the target data root directory {} is not empty! ".format(self.data_root) +
                             "\nType in \"yes\" to overwrite everything and continue (yes/no)\n")
            assert response in ["yes", "y", "Y"], "please empty the target data root directory or " \
                                                  "change the data_root setting."

    def _get_valid_slice(self, slice_start=None, slice_end=None):
        # 注意，返回的slice是左闭右开，即包含start_index，不包括end_index
        slice_start = self._save_target_slice[0] if slice_start is None else slice_start
        slice_end = self._save_target_slice[1] if slice_end is None else slice_end
        # slice_start = 0 if slice_start is None else slice_start
        # slice_end = len(self) if slice_end is None else slice_end
        # assert 0 <= slice_start <= slice_end <= len(self), "invalid slice range"
        slice_start = max([0, slice_start])
        slice_end = min([slice_end, len(self)])
        slice_end = max([slice_end, slice_start]) # 0 <= slice_start <= slice_end <= len(self)
        return slice_start, slice_end

    def _extend_target_slice(self, add_length=1):
        '''
        在队列尾部store新的数据时，更新索引
        目前不考虑在队列头部或中间store新的数据的情况
        '''
        if self._save_target_slice[1] < len(self): # 理论上_save_target的end都是顶到队列末端的
            print("DEBUG: no need to extend target slice")
            return

        if len(self) + add_length > self.maxlen:
            delta = len(self) + add_length - self.maxlen
            self._save_target_slice[1] = self.maxlen
            self._save_target_slice[0] -= delta
            self._save_target_slice[0] = max([0, self._save_target_slice[0]])
        else:
            self._save_target_slice[1] += 1

    def _autosave_check(self, done):
        '''检查是否需要自动保存到本地数据集'''
        if self._autosave:
            if self.enough_saves():
                print("NOTICE: Buffer has already saved enough experiences ({} records). ".format(self._count_saves) +
                      "Will no more autosave offline dataset.")
                return False

            if self._autosave_episode_done and done:
                return True
            idx1, idx2 = self._save_target_slice
            if idx2 - idx1 >= self._autosave_max_intervals:
                return True
        return False

    def _clear_record(self):
        self._temp_record = {
            # "timestamp": None,
            "forward": None,
            "feedback": None
        }

    # def _is_waiting_feedback(self) -> bool:
    #     # 1. 要求既存forward也存feedback
    #     # 2. 要求已经存在已有的forward
    #     # 3. 还未存forward对应的feedback
    #     return (not self._STORE_FORWARD_ONLY) and \
    #            (self._temp_record["forward"] is not None) and \
    #            (self._temp_record["feedback"] is None)

    ################# modify some functions that changes the buffer content #################
    def append(self, __x) -> None:
        self._extend_target_slice(1)
        super(BaseBuffer, self).append(__x)

    # def appendleft(self, __x) -> None:
    #     self._save_target_slice = self._get_valid_slice(
    #         self._save_target_slice[0]+1, self._save_target_slice[1]+1)
    #     super(BaseBuffer, self).appendleft(__x)

    def insert(self, __i: int, __x) -> None: # 左端插入认为无视，相应slice只需要往后延
        start, end = self._save_target_slice
        if start > __i:
            start += 1
        if end > __i:
            end += 1
        self._save_target_slice = self._get_valid_slice(start, end)
        super(BaseBuffer, self).insert(__i, __x)

    def extend(self, __iterable) -> None:
        self._extend_target_slice(len(__iterable))
        super(BaseBuffer, self).extend(__iterable)



class LogBuffer(BaseBuffer):
    """
    Buffer for storing log data.
    [timestamp, observation, plan, reward, done]
    """
    def __init__(self,
                 maxlen=_default_max_len,
                 # 保存到本地数据集的参数
                 data_root='./dataset/',
                 file_prefix='log',
                 file_format=spider.DATASET_FORMAT_JSON,
                 # 下面是自动保存离线数据集时候的参数
                 autosave=True,  # 只有autosave为True时，以下参数才有意义
                 autosave_max_intervals=1000,
                 autosave_episode_done=True,  # done信号了要不要保存一次数据到本地, 如果不保存就是固定intervals
                 max_saves=_default_max_saves,
                 ):
        super(LogBuffer, self).__init__(maxlen, data_root,file_prefix,file_format,autosave,
                                        autosave_max_intervals,autosave_episode_done,max_saves)
        # self: Deque[float, elm.Observation, elm.Plan, float, bool]

    # def record_forward(self, timestamp, observation:elm.Observation, plan:elm.Plan):
    #     self._temp_record["forward"] = [timestamp, observation, plan]
    #
    #     if self._STORE_FORWARD_ONLY:
    #         # _STORE_FORWARD_ONLY若为True，则record_forward后自动存入buffer，清除record
    #         self.store(timestamp, observation, plan)
    #         self._clear_record()
    #     else:
    #         pass # wait for record_feedback...
    #
    # def record_feedback(self, reward, done):
    #     if self._temp_record["forward"] is None:
    #         return # 只有在有forward信息时才存feedback信息
    #
    #     self._temp_record["feedback"] = [reward, done]
    #
    #     if not self._STORE_FORWARD_ONLY:
    #         # _STORE_FORWARD_ONLY若为True，则record_forward后自动存入buffer，清除record
    #         self.store(*self._temp_record["forward"], reward, done)
    #         self._clear_record()

    def store(self, timestamp, observation:elm.Observation, plan:elm.Plan,
              reward:float=None, done:bool=None, *other_args):
        # self._extend_target_slice()
        self.append((timestamp, observation, plan, reward, done, *other_args))

        if self._autosave_check(done):
            self.save()


    def save(self, filepath=None, slice_start=None, slice_end=None):
        '''
        如果不加参数，则自动保存当前target_slice内的数据。数据名默认往后递推
        如果有参数，则保存指定范围内的数据。数据名参照给定的
        '''
        filepath = self._get_filepath(self._count_saves) if filepath is None else filepath
        slice_start, slice_end = self._get_valid_slice(slice_start, slice_end)
        # file_format = self.file_format if file_format is None else file_format

        if self.file_format == spider.DATASET_FORMAT_JSON:
            self.save_json(filepath, slice_start, slice_end)
        elif self.file_format == spider.DATASET_FORMAT_RAW:
            self.save_raw(filepath, slice_start, slice_end)
        else:
            raise ValueError("unsupported dataset format")

        print("Successfully saved {} log records to {}".format(slice_end-slice_start, filepath))
        self._count_saves += 1
        self._save_target_slice = [slice_end, slice_end]


    def replay(self):
        pass


    def apply_to(self, spider_planner:spider.planner_zoo.BasePlanner, spider_reward_model=None): # 用decorator监听模块
        spider_planner._activate_log_buffer = True
        spider_planner._log_buffer = self
        spider_planner.plan = logbuffer_plan(spider_planner.plan) # wrapper装饰器 非常核心！！！
        self._STORE_FORWARD_ONLY = True
        print("LogBuffer: Log Buffer is listening to the planner.")

        if spider_reward_model is not None:
            # todo: spider_reward_model的监听
            self._STORE_FORWARD_ONLY = False
            raise NotImplementedError("Decorators for reward model has not been implemented...")

        print("LogBuffer: All data will be automatically recorded...")


    def to_experience_buffer(self, state_encoder, action_encoder):
        pass


    def save_json(self, filepath, slice_start, slice_end):
        # slice_start, slice_end = self._get_valid_slice(slice_start, slice_end)

        data = {}
        for i in range(slice_start, slice_end):
            assert len(self[i]) == 5, \
                "To save standard json file, Log data must have records that only contains 5 elements, " \
                "which is timestamp, observation, plan, reward, done.\n" \
                "If you want to save more information, use save_raw instead!"

            timestamp, (ego_state, perception, local_map), plan, reward, done = self[i]
            assert isinstance(perception, spider.elements.TrackingBoxList), \
                "perception must be TrackingBoxList for now"  # 暂时没有设计occupancy的支持

            if timestamp is None:
                timestamp = i

            data[timestamp] = {
                "ego_state": ego_state.to_dict() if ego_state is not None else None,
                "perception": [tb.to_dict() for tb in perception] if perception is not None else None,
                "local_map": local_map.to_dict() if local_map is not None else None,
                "plan": plan.to_dict() if plan is not None else None,
                "reward": reward,
                "done": done
            }

        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


    def save_raw(self, filepath, slice_start, slice_end):
        if slice_start==0 and slice_end==len(self):
            target = list(self)
        else:
            slice_start, slice_end = self._get_valid_slice(slice_start, slice_end)
            target = self.get_list_by_slice(slice_start, slice_end)

        with open(filepath, "wb") as file:
            pickle.dump(target, file)




class ExperienceBuffer(BaseBuffer):
    """
    Buffer for storing experience data.
    [timestamp, state, action, next_state, reward, done]
    # exp buffer一般不存时间戳？
    # 注意！！！ state action等等，在存入exp buffer的时候，不应该是batched

    """
    def __init__(self,
                 maxlen=_default_max_len,
                 # 保存到本地数据集的参数
                 data_root='./dataset/',
                 file_prefix='exp',
                 file_format=spider.DATASET_FORMAT_TENSOR,
                 # 下面是自动保存离线数据集时候的参数
                 autosave=True,  # 只有autosave为True时，以下参数才有意义
                 autosave_max_intervals=1000,
                 autosave_episode_done=True,  # done信号了要不要保存一次数据到本地, 如果不保存就是固定intervals
                 max_saves=_default_max_saves,
                 ):
        super(ExperienceBuffer, self).__init__(maxlen, data_root,file_prefix,file_format,autosave,
                                               autosave_max_intervals,autosave_episode_done,max_saves)

    def store(self, timestamp, state:torch.Tensor, action:torch.Tensor, reward:float=None, done:bool=None):
        # next_state的处理，目前是每次只记录state,action,reward,done以及None
        # 在下一次储存的时候，会检查上一条记录，如果done，则不操作；否则将上一条记录的next_state改为当前state
        # todo:在当前监听记录的逻辑下，每个episode的最后一个记录，next_state的值是None，因为已经done了，不会再监听到下一条记录，所以无法更新next_state
        # todo:另外，如果环境或reward function没有提供done信息，目前的next_state逻辑是错误的，因为无法判断什么时候应该结束一个episode

        #将输入都转到cpu，同时detach
        # 要不要clone？怕引用赋值有问题
        timestamp, state, action, reward, done = self._to_cpu(timestamp, state, action, reward, done)

        if len(self) > 0: # 若存在上一条记录
            last_record = self[-1]
            if not last_record[4]: #上一条记录的done是False，说明不是一个episode的结束，当前record是上一条record的next
                last_record[5] = state # 将当前state作为上一条record的next_state

        # self._extend_target_slice(1)
        next_state = None
        self.append([timestamp, state, action, reward, done, next_state])

        if self._autosave_check(done):
            self.save()

    def save(self, filepath=None, slice_start=None, slice_end=None):
        '''
        如果不加参数，则自动保存当前target_slice内的数据。数据名默认往后递推
        如果有参数，则保存指定范围内的数据。数据名参照给定的
        '''
        filepath = self._get_filepath(self._count_saves) if filepath is None else filepath
        slice_start, slice_end = self._get_valid_slice(slice_start, slice_end)
        # file_format = self.file_format if file_format is None else file_format

        if self.file_format == spider.DATASET_FORMAT_TENSOR:
            self.save_tensor(filepath, slice_start, slice_end)
        elif self.file_format == spider.DATASET_FORMAT_RAW:
            self.save_raw(filepath, slice_start, slice_end)
        else:
            raise ValueError("unsupported dataset format")

        print("Successfully saved {} EXP records to {}".format(slice_end - slice_start, filepath))
        self._count_saves += 1
        self._save_target_slice = [slice_end, slice_end]


    def apply_to(self, policy:nn.Module, spider_reward_model=None): # 用decorator监听模块
        policy._activate_exp_buffer = True
        policy._exp_buffer = self
        policy.forward = expbuffer_policy(policy.forward) # wrapper装饰器 非常核心！！！
        self._STORE_FORWARD_ONLY = True
        print("ExperienceBuffer: EXP Buffer is listening to the policy.")


        if spider_reward_model is not None:
            # todo: spider_reward_model的监听
            self._STORE_FORWARD_ONLY = False
            raise NotImplementedError("Decorators for reward model has not been implemented...")

        print("ExperienceBuffer: All data will be automatically recorded...")
        print("ExperienceBuffer: Please notice that when logging experience, all data should NOT be batched.")



    def save_tensor(self, filepath, slice_start, slice_end):
        # qzl:笑死，看了一下torch文档，torch.save就是用的pickle，二者没区别。。。
        if slice_start == 0 and slice_end == len(self):
            target = list(self)
        else:
            slice_start, slice_end = self._get_valid_slice(slice_start, slice_end)
            target = self.get_list_by_slice(slice_start, slice_end)

        target = [self._to_tensor(*record) for record in target]
        torch.save(target, filepath)


    def save_raw(self, filepath, slice_start, slice_end):
        if slice_start == 0 and slice_end == len(self):
            target = list(self)
        else:
            slice_start, slice_end = self._get_valid_slice(slice_start, slice_end)
            target = self.get_list_by_slice(slice_start, slice_end)

        with open(filepath, "wb") as file:
            pickle.dump(target, file)

    def replay(self):
        pass

    def get_dataloader(self):
        pass

    def _to_cpu(self,*args):
        cpu_args = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args]

        if len(args) == 1:
            return cpu_args[0]
        else:
            return cpu_args

    def _to_tensor(self, *args):
        numerical_args = [0.0 if x is None else x for x in args]
        tensor_args = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in numerical_args]

        if len(args) == 1:
            return tensor_args[0]
        else:
            return tensor_args
