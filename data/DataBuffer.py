import itertools
import warnings
from typing import Deque, Tuple
import random
from collections import deque
from abc import abstractmethod
import os

import torch
import torch.nn as nn


import spider
import spider.elements as elm
from spider.data.decorators import *
from spider.data.data_factory import *

_default_max_len = 10000
_default_max_records = 10000

# todo：以后加一个更加通用的databuffer，以及其装饰器。存任意形式的数据，并且保存到本地数据集。支持任意形式的函数的输入



class BaseBuffer(deque):
    def __init__(self,
                 maxlen=_default_max_len,
                 forward_only=True,
                 # 保存到本地数据集的参数
                 data_root='./dataset/',
                 subdir_prefix='segment',
                 file_format=spider.DATA_FORMAT_RAW,
                 # 下面是自动保存离线数据集时候的参数
                 autosave=True,  # 只有autosave为True时，以下参数才有意义
                 autosave_max_intervals=1000,
                 new_seg_when_done=True,  # done信号了要不要保存一次数据到本地, 如果不保存就是固定intervals
                 max_save_records=_default_max_records,
                 ):

        super(BaseBuffer, self).__init__(maxlen=maxlen)

        self._autosave: bool = autosave
        self._new_seg_when_done: bool = new_seg_when_done  # done信号了要不要保存一次数据到本地
        self._autosave_max_intervals: int = autosave_max_intervals
        assert autosave_max_intervals <= maxlen, "autosave_max_intervals should not be larger than maxlen"

        self.data_root = data_root
        self.sub_dir:str = None
        self._sub_dir_prefix = subdir_prefix
        self.file_format = file_format

        self.seg_idx = 0 # 当前的segment record_index
        self.record_idx = 0 # 在当前的segment中，保存到了第几条record
        self.count_records = 0 # 总共保存了多少条record
        self.max_records = int(max_save_records)

        self._subdir_index_length = len(str(self.max_records - 1))  # 判断几位数的方式好像不太好
        self._filename_index_length = len(str(self._autosave_max_intervals - 1))


        self._save_target_slice = [0, 0] # 记录当前episode的起始和结束位置(左闭右开)

        self._temp_record = { # 暂时储存一条数据，存满了就放入buffer
            # "timestamp": None,
            "forward": None,
            "feedback": None
        }
        self._STORE_FORWARD_ONLY = forward_only
        # self._STORE_FORWARD_FEEDBACK = False
        # _STORE_FORWARD_ONLY若为True，则record_forward后自动存入buffer，清除record
        # _STORE_FORWARD_ONLY若为False，则record完毕forward后，等待record_feedback后再自动存入buffer，清除record

        if self._autosave:
            self._prepare_data_root()
            self._update_sub_dir(self.seg_idx)  # 从0开始创建子文件夹

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
    def replay(self, *args, **kwargs):
        # todo:想清楚replay函数到底是想实现什么功能。
        #  0406：一个新的想法！！replay应该选定某一个或某几个片段。返回一个迭代器/生成器，一直返回某个离线的log片段
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
        # if self._autosave and not self.enough_saves():
        #     if self._save_target_slice[1] - self._save_target_slice[0]>0:
        #         self.save()
        self.clear()


    def shuffle(self):
        random.shuffle(self)
        self._save_target_slice = [len(self), len(self)]

    def get_list_by_slice(self, start, end):
        return list(itertools.islice(self, start, end))

    def enough_saves(self) -> bool:
        # 检查是否已经保存了足够的数据，如果达到阈值，则不再保存
        return self.count_records >= self.max_records


    def _update_sub_dir(self, seg_idx):
        self.seg_idx = seg_idx
        sub_dir_name = self._sub_dir_prefix + str(seg_idx).zfill(self._subdir_index_length)
        self.sub_dir = os.path.join(self.data_root, sub_dir_name)
        self.record_idx = 0

        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)

    def _get_filepath(self, record_idx):
        filename = str(record_idx).zfill(self._subdir_index_length) \
                   + format_suffix[self.file_format]
        return os.path.join(self.sub_dir, filename)

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

    def _segment_end(self, done):
        '''
        :param done: 是否结束当前episode
        :return: 是否需要分割segment
        '''
        if self._new_seg_when_done and done:
            return True
        if self.record_idx >= self._autosave_max_intervals:
            return True
        return False

    def _autosave_check(self):
        '''检查是否需要自动保存到本地数据集'''
        if self._autosave:
            if self.enough_saves():
                print("NOTICE: Buffer has already saved enough data ({} records). ".format(self.count_records) +
                      "Will no more autosave offline dataset.")
                return False
            else:
                return True
        else:
            return False

            # if self._autosave_episode_done and done:
            #     return True
        #     idx1, idx2 = self._save_target_slice
        #     if idx2 - idx1 >= self._autosave_max_intervals:
        #         return True
        # return False

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
                 forward_only=True,
                 # 保存到本地数据集的参数
                 data_root='./dataset/',
                 subdir_prefix='log_segment',
                 file_format=spider.DATA_FORMAT_JSON,
                 # 下面是自动保存离线数据集时候的参数
                 autosave=True,  # 只有autosave为True时，以下参数才有意义
                 autosave_max_intervals=1000,
                 new_seg_when_done=True,  # done信号了要不要保存一次数据到本地, 如果不保存就是固定intervals
                 max_save_records=_default_max_records,
                 ):
        super(LogBuffer, self).__init__(maxlen,forward_only, data_root, subdir_prefix, file_format, autosave,
                                        autosave_max_intervals, new_seg_when_done, max_save_records)
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

        if self._autosave_check():
            #  后面这个逻辑修改了一下，现在是每次有数据进来都保存，episode结束的时候造新的子文件夹,
            #  以前的逻辑是一个episode结束了才保存
            self.save()
            if self._segment_end(done):
                print("A segment containing {} log records has been saved to {}".format(self.record_idx, self.sub_dir))
                self._update_sub_dir(self.seg_idx + 1)



    def save(self, filepath=None):
        '''
        保存buffer中最后一个的数据
        如果不加参数，数据名默认取record idx
        如果有参数，则数据名参照给定的
        '''
        filepath = self._get_filepath(self.record_idx) if filepath is None else filepath

        if self.file_format == spider.DATA_FORMAT_JSON:
            save_json_log(filepath, self[-1])
        elif self.file_format == spider.DATA_FORMAT_RAW:
            save_raw(filepath, self[-1])
        else:
            raise ValueError("unsupported dataset format")

        # print("Successfully saved {} log records to {}".format(slice_end-slice_start, filepath))
        self.count_records += 1
        self.record_idx += 1


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






class ExperienceBuffer(BaseBuffer):
    """
    Buffer for storing experience data.
    [timestamp, state, action, reward, done, next_state]
    # exp buffer一般不存时间戳？
    # 注意！！！ state action等等，在存入exp buffer的时候，不应该是batched

    """
    def __init__(self,
                 maxlen=_default_max_len,
                 forward_only=False,
                 # 保存到本地数据集的参数
                 data_root='./dataset/',
                 subdir_prefix='exp_segment',
                 file_format=spider.DATA_FORMAT_TENSOR,
                 # 下面是自动保存离线数据集时候的参数
                 autosave=False,  # 只有autosave为True时，以下参数才有意义
                 autosave_max_intervals=1000,
                 new_seg_when_done=True,  # done信号了要不要保存一次数据到本地, 如果不保存就是固定intervals
                 max_save_records=_default_max_records,
                 ):
        super(ExperienceBuffer, self).__init__(maxlen,forward_only, data_root, subdir_prefix, file_format, autosave,
                                               autosave_max_intervals, new_seg_when_done, max_save_records)

    def store(self, state:torch.Tensor, action:torch.Tensor, reward:float=None, done:bool=None):
        # next_state的处理，目前是每次只记录state,action,reward,done以及None
        # 在下一次储存的时候，会检查上一条记录，如果done，则不操作；否则将上一条记录的next_state改为当前state
        # todo:在当前监听记录的逻辑下，每个episode的最后一个记录，next_state的值是state，
        #  因为已经done了，不会再监听到下一条记录，所以无法更新next_state。不过done了的话本身也用不到next_state,所以没事
        # todo:另外，如果环境或reward function没有提供done信息，目前的next_state逻辑是错误的，因为无法判断什么时候应该结束一个episode

        #将输入都转到cpu，同时detach
        # 要不要clone？怕引用赋值有问题
        state, action, reward, done = to_tensor(state, action, reward, done)
        state, action, reward, done = ensure_batched(state, action, reward, done)
        state, action, reward, done = to_cpu(state, action, reward, done)

        self.try_update_last_record(state) # 尝试将当前state作为上一条record的next_state

        # self._extend_target_slice(1)
        next_state = state # 暂时记录当前的state，下一步会try_update_last_record更新这个记录的值
        self.append([state, action, reward, done, next_state])

        if self._autosave_check():
            self.save()
            if self._segment_end(done.item()):
                print("A segment containing {} exp records has been saved to {}".format(self.record_idx, self.sub_dir))
                self._update_sub_dir(self.seg_idx + 1)

    def save(self, filepath=None):
        '''
        保存buffer中最后一个的数据
        如果不加参数，数据名默认取record idx
        如果有参数，则数据名参照给定的
        '''

        if self.file_format == spider.DATA_FORMAT_TENSOR:
            save_tensor(filepath, self[-1])
        elif self.file_format == spider.DATA_FORMAT_RAW:
            save_raw(filepath, self[-1])
        else:
            raise ValueError("unsupported dataset format")

        self.count_records += 1
        self.record_idx += 1


    def apply_to(self, policy:nn.Module, spider_reward_model=None): # 用decorator监听模块
        policy._activate_exp_buffer = True
        policy._exp_buffer = self
        policy.forward = expbuffer_policy(policy.forward) # wrapper装饰器 非常核心！！！
        self._STORE_FORWARD_ONLY = True
        print("ExperienceBuffer: EXP Buffer is listening to the policy. All forward data will be recorded...")


        if spider_reward_model is not None:
            # spider_reward_model的监听
            self._STORE_FORWARD_ONLY = False
            spider_reward_model._activate_exp_buffer = True
            spider_reward_model._exp_buffer = self
            spider_reward_model.evaluate_log = expbuffer_reward(spider_reward_model.evaluate_log)
            print("ExperienceBuffer: EXP Buffer is listening to the reward. All feedback data will be recorded...")
            # evaluate_exp暂时不加

        print("ExperienceBuffer: Please notice that when logging experience, all data SHOULD be batched.")


    def sample(self, batch_size:int, device=None):
        '''
        return batched tensor of experiences
        '''
        batch_size = min([len(self), batch_size])
        if batch_size <= 0 :
            warnings.warn("batch size or the experience buffer length is 0. Ignore sampling")
            return None

        batched_info = random.sample(self, batch_size)

        states, actions, rewards, dones, next_states = [torch.cat(x, dim=0) for x in zip(*batched_info)]

        # if batch_size == 1:
        #     # batched
        #     states, actions, rewards, dones, next_states = [torch.cat(x, dim=0) for x in zip(*batched_info)]
        # else:
        #     states, actions, rewards, dones, next_states = [torch.cat(x, dim=0) for x in zip(*batched_info)]
        return states, actions, rewards, dones, next_states

    def replay(self):
        pass

    def get_dataloader(self):
        pass

    def try_update_last_record(self, current_state):
        '''
        如果存在上一条记录，并且上一条记录的done是False，则更新上一条记录的next_state为当前state
        '''
        if len(self) > 0:  # 若存在上一条记录
            last_record = self[-1]
            if not last_record[3]:  # 上一条记录的done是False，说明不是一个episode的结束，当前record是上一条record的next
                last_record[4] = current_state  # 将当前state作为上一条record的next_state


