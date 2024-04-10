from typing import Union, Sequence
import warnings
import tqdm
import os

import torch
from torch.utils.data import Dataset as torch_Dataset

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


class OnlineDataset(torch_Dataset):
    '''
    从在线的经验池创建Dataset
    '''
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class OfflineExpDataset(torch_Dataset):
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
        # self._use_exp_buffer = use_exp_buffer

        # self.record_nums = get_record_num_for_segments(data_root)

        # todo: 以下是建立index到文件名的映射。
        #  这里需要优化，因为一旦文件多的话扫描很慢，映射也会变慢, 而且占内存。可以考虑换成segment idx&record record_index
        self._log_range = {}  # log_id : [start_record_id, end_record_id]
        self._record_id2file = {} # record id to the record file name, 左闭右开
        record_id = 0
        log_id = 0
        for sub_dir_name in tqdm.tqdm(os.listdir(data_root), desc="Scanning all the sub directories"):
            sub_dir = os.path.join(self.data_root, sub_dir_name)
            if not os.path.isdir(sub_dir):
                print("Non-Directory is detected in the data_root. Ignoring {}".format(sub_dir))
                continue


            start_record_id = record_id

            for filename in os.listdir(sub_dir):
                # 注意, listdir也会把文件夹包含在内，这里是默认seg里面不包含文件夹
                filepath = os.path.join(sub_dir, filename)
                if not os.path.isfile(filepath):
                    print("Non-File is detected in the data_root. Ignoring {}".format(filepath))
                    continue
                self._record_id2file[record_id] = filepath
                record_id += 1

            self._log_range[log_id] = [start_record_id, record_id]
            log_id += 1


    def __len__(self):
        return len(self._record_id2file)

    def __getitem__(self, index):
        filepath = self._record_id2file[index]
        raise NotImplementedError


class OfflineLogDataset(OfflineExpDataset):
    '''
    从离线的Log数据集文件夹创建exp Dataset
    '''
    def __init__(self, data_root, state_encoder, action_encoder,
                 file_format=spider.DATA_FORMAT_RAW, groundtruth=spider.DATA_GT_PLAN,
                 require_feedback=False, require_next_state=False):
        super(OfflineLogDataset, self).__init__(data_root, file_format, groundtruth, require_feedback, require_next_state)
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

    def __len__(self):
        # return 1
        return len(self._record_id2file)

    def __getitem__(self, record_index):
        filepath = self._record_id2file[record_index]
        log_record = self._load_data(filepath)
        state = self.state_encoder(*log_record[1])
        action = self._get_action(log_record)

        reward = done = torch.empty((0,))  # reward和done
        if self.require_feedback:
            reward = to_tensor(log_record[3]) if log_record[3] is not None else None
            done = to_tensor(log_record[4]) if log_record[4] is not None else None

        next_state = torch.empty((0,))  # next_state
        if self.require_next_state:
            if record_index + 1 > self.__len__() or log_record[4]: # 如果索引超界了，或者当前记录就是终止状态(done)
                next_state = torch.empty((0,))
            else:
                next_state = self._get_next_state(record_index)

        return state, action, reward, done, next_state

    def get_dataloader(self, *args, **kwargs):
        '''batch_size = 50,shuffle = False,sampler=None,batch_sampler = None,num_workers = 0,
        collate_fn = pin_memory = False,drop_last = False,timeout = 0,worker_init_fn = None,'''
        return torch.utils.data.DataLoader(self, *args, **kwargs)

    def get_record_by_id(self, record_index):
        filepath = self._record_id2file[record_index]
        log_record = self._load_data(filepath)
        return log_record

    def get_records_by_log(self, log_index, record_iterator=False):
        '''
        log_index:
        record_iterator: return an interator or a log buffer? by default False
        '''
        from spider.data.DataBuffer import LogBuffer

        start_id, end_id = self._log_range[log_index]

        if record_iterator:
            # 返回Iterator节省内存开销以及初始化时的运算开销，加大每次迭代的时候的运算开销
            def _record_iterate():
                for record_id in range(start_id, end_id):
                    yield self.get_record_by_id(record_id)
            return _record_iterate()
        else:
            # 返回LogBuffer的话，需要初始化整个LogBuffer，然后返回，初始化时的开销大，但是每次迭代时开销小
            log_buffer = LogBuffer(maxlen=None, autosave=False)
            for record_id in range(start_id, end_id):
                log_record = self.get_record_by_id(record_id)
                log_buffer.store(*log_record)
            return log_buffer

    def replay(self, spider_planner:spider.planner_zoo.BasePlanner, log_ids: Union[int, Sequence[int]],
               max_steps=None, ego_veh_replay=True, recording=False):
        '''
        log replay
        ego_veh_replay: if False, then ego vehicle will execute the planner's output. Otherwise, it will replay the ego trace
        '''
        log_ids = [log_ids] if isinstance(log_ids, int) else log_ids

        import spider.visualize as vis
        from spider.elements import VehicleState
        ego_len, ego_wid = spider_planner.length, spider_planner.width

        vis.figure(figsize=(12, 8))
        if recording:
            snapshot = vis.SnapShot(False, record_video=True, video_path='log_replay.avi', fps=15)

        count = 0
        for log_id in log_ids:
            records = self.get_records_by_log(log_id, record_iterator=True)
            _next_state = None
            for record in records:
                t, (ego_state, perception, local_map), gt_traj = record[:3]

                if (not ego_veh_replay) and (_next_state is not None):
                    ego_state = _next_state

                planned_traj = spider_planner.plan(ego_state, perception, local_map)

                if not ego_veh_replay:
                    # 轨迹执行器暂时是完美控制，直接到达下一个点
                    # 以后要修改：第一，不应该是完美控制，应该由控制器控制
                    # 第二，不应该是下一个点。应该根据时间戳的差值，来计算执行了几步。当然如果直接上了控制器就不用管了
                    _next_state = VehicleState.from_traj_step(planned_traj, 1, ego_len, ego_wid)

                vis.cla()
                if gt_traj is not None:
                    vis.draw_trajectory(gt_traj, color="C0", show_footprint=True, footprint_alpha=0.2, label="groundtruth")
                vis.lazy_draw(ego_state, perception, local_map, planned_traj)
                vis.pause(0.01)

                if recording:
                    snapshot.snap()

                count += 1
                if (max_steps is not None) and (count >= max_steps):
                    break

        vis.close()
        # vis.show()


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
        filepath = self._record_id2file[current_index + 1]
        next_log_record = self._load_data(filepath)
        next_state = self.state_encoder(next_log_record[1])
        return next_state


if __name__ == '__main__':
    from spider.planner_zoo.MlpPlanner import MlpPlanner
    # setup the planner
    planner = MlpPlanner({})

    dataset = OfflineLogDataset('../DEBUGtest/dataset/', planner.state_encoder, planner.action_encoder)


