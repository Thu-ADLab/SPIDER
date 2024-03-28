import pickle
import json


import spider



class DataFactory:
    """
    Data factory to convert different data format.

    log_buffer: [ego_state, perception, local_map, action, reward, done]

    save_xxx():
    将 log_buffer 以xxx格式存储到本地。

    load_xxx():
    将本地的xxx格式数据读取出来，返回log_buffer。

    get_dataloader_xxx():
    根据xxx对象生成dataloader。
    若xxx为buffer，则直接返回在线的log_buffer建立的dataloader
    否则，需要建立dataloader将xxx格式的数据从data_root中读取出来。

    # todo: 想办法统一一下现在零碎的方法，比如用save_log（）作为统一接口
    # todo: 思考一下，是直接写成函数，还是封装成类？要不要改为静态方法？


    """
    def __init__(self, batch_size, device):

        # 都是data_loader要用的参数
        self.batch_size = batch_size
        self.device = device # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def save_log(self, log_buffer, data_format=spider.DATASET_FORMAT_JSON, *args, **kwargs):
    #     if data_format == spider.DATASET_FORMAT_JSON:
    #         return self.save_json(log_buffer, *args, **kwargs)

    def save_json(self, filepath, log_buffer, keys):
        '''
        Save logs in json format.

        Parameters:
            log_buffer (list): A list of logs.
            keys (list): Keys for the logs. 指定一些keys记录。未来完成
        '''

        data = {}
        for timestamp, ego_state, perception, local_map, action, reward, done in log_buffer:
            assert isinstance(perception, spider.elements.TrackingBoxList), \
                "perception must be TrackingBoxList for now" # 暂时没有设计occupancy的支持

            data[timestamp] = {
                "ego_state": ego_state.to_dict(),
                "perception": [tb.to_dict() for tb in perception],
                "local_map": local_map.to_dict(),
                "action": action.to_dict(),
                "reward": reward,
                "done": done
            }
            # with open(filepath, "a") as file:
            #     file.write(f"{ego_state}, {perception}, {local_map}, {action}, {reward}, {done}\n")

    def save_raw(self, filepath, log_buffer):
        with open(filepath, "wb") as file:
            pickle.dump(log_buffer, file)

    def save_tensor(self, filepath, log_buffer, state_encoder, action_decoder):
        pass

    def load_json(self, filepath):
        pass

    def load_raw(self, filepath):
        with open(filepath, "rb") as file:
            log = pickle.load(file)
        return log

    def load_tensor(self):
        pass



    def get_dataloader_json(self, data_root):
        pass

    def get_dataloader_raw(self, data_root):
        pass

    def get_dataloader_tensor(self, data_root):
        pass

    def get_dataloader_buffer(self, log_buffer):
        '''在线构建Buffer的dataloader，实际上跟经验回放池类似'''
        pass
