from typing import Sequence, TypeVar
import pickle
import json
import torch

import spider

Log = TypeVar("Log")

format_suffix = {
    spider.DATA_FORMAT_JSON: '.json',
    spider.DATA_FORMAT_RAW: '.pkl',
    spider.DATA_FORMAT_TENSOR: '.pt'
} # todo: 以后可以考虑支持numpy的npz格式 或者 csv 格式


def to_cpu(*args):
    cpu_args = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in args]

    if len(args) == 1:
        return cpu_args[0]
    else:
        return cpu_args

def to_tensor(*args, dtype=torch.float32):
    numerical_args = [0.0 if x is None else x for x in args]
    tensor_args = [x if isinstance(x, torch.Tensor) else torch.tensor(x,dtype=dtype) for x in numerical_args]

    if len(args) == 1:
        return tensor_args[0]
    else:
        return tensor_args

def save_json(filepath, data_dict: dict):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)

def save_json_log(filepath, log_record:Log):
    assert len(log_record) == 5, \
        "To save standard json file, Log data must have records that only contains 5 elements, " \
        "which is timestamp, observation, plan, reward, done.\n" \
        "If you want to save more information, use save_raw instead!"

    timestamp, (ego_state, perception, local_map), plan, reward, done = log_record
    assert isinstance(perception, spider.elements.TrackingBoxList), \
        "perception must be TrackingBoxList for now"  # 暂时没有设计occupancy的支持

    data = {
        "timestamp": timestamp,
        "ego_state": ego_state.to_dict() if ego_state is not None else None,
        "perception": [tb.to_dict() for tb in perception] if perception is not None else None,
        "local_map": local_map.to_dict() if local_map is not None else None,
        "plan": plan.to_dict() if plan is not None else None,
        "reward": reward,
        "done": done
    }

    save_json(filepath, data)



def save_raw(filepath, data):
    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def save_tensor(filepath, data_sequence: Sequence):
    # qzl:笑死，看了一下torch文档，torch.save就是用的pickle，二者没区别。。。

    data = to_tensor(*data_sequence)
    torch.save(data, filepath)

def load_raw(filepath):
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    return data

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_tensor(filepath):
    data = torch.load(filepath)
    return data


# class DataFactory:
#     """
#     Data factory to convert different data format.
#
#     log_buffer: [ego_state, perception, local_map, action, reward, done]
#
#     save_xxx():
#     将 log_buffer 以xxx格式存储到本地。
#
#     load_xxx():
#     将本地的xxx格式数据读取出来，返回log_buffer。
#
#     get_dataloader_xxx():
#     根据xxx对象生成dataloader。
#     若xxx为buffer，则直接返回在线的log_buffer建立的dataloader
#     否则，需要建立dataloader将xxx格式的数据从data_root中读取出来。
#
#
#     """
#     def __init__(self, batch_size, device):
#
#         # 都是data_loader要用的参数
#         self.batch_size = batch_size
#         self.device = device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # def save_log(self, log_buffer, data_format=spider.DATASET_FORMAT_JSON, *args, **kwargs):
#     #     if data_format == spider.DATASET_FORMAT_JSON:
#     #         return self.save_json(log_buffer, *args, **kwargs)
#
#     def save_json(self, filepath, log_buffer, keys):
#         '''
#         Save logs in json format.
#
#         Parameters:
#             log_buffer (list): A list of logs.
#             keys (list): Keys for the logs. 指定一些keys记录。未来完成
#         '''
#
#         data = {}
#         for timestamp, ego_state, perception, local_map, action, reward, done in log_buffer:
#             assert isinstance(perception, spider.elements.TrackingBoxList), \
#                 "perception must be TrackingBoxList for now" # 暂时没有设计occupancy的支持
#
#             data[timestamp] = {
#                 "ego_state": ego_state.to_dict(),
#                 "perception": [tb.to_dict() for tb in perception],
#                 "local_map": local_map.to_dict(),
#                 "action": action.to_dict(),
#                 "reward": reward,
#                 "done": done
#             }
#             # with open(filepath, "a") as file:
#             #     file.write(f"{ego_state}, {perception}, {local_map}, {action}, {reward}, {done}\n")
#
#     def save_raw(self, filepath, log_buffer):
#         with open(filepath, "wb") as file:
#             pickle.dump(log_buffer, file)
#
#     def save_tensor(self, filepath, log_buffer, state_encoder, action_decoder):
#         pass
#
#     def load_json(self, filepath):
#         pass
#
#     def load_raw(self, filepath):
#         with open(filepath, "rb") as file:
#             log = pickle.load(file)
#         return log
#
#     def load_tensor(self):
#         pass
#
#
#
#     def get_dataloader_json(self, data_root):
#         pass
#
#     def get_dataloader_raw(self, data_root):
#         pass
#
#     def get_dataloader_tensor(self, data_root):
#         pass
#
#     def get_dataloader_buffer(self, log_buffer):
#         '''在线构建Buffer的dataloader，实际上跟经验回放池类似'''
#         pass
