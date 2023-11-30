import random
from collections import deque
import pickle


class ExperienceBuffer:
    def __init__(self, buffer_size=100000):
        self.max_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = [None, None, None, None, None]

    @property
    def size(self):
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer.clear()

    def clear_experience(self):
        self.experience = [None, None, None, None, None]

    def store_experience(self, experience):
        # experience = (state, action, reward, next_state, done)
        if self.size >= self.max_size:
            print("Experience Replay Buffer already full!")

        if all(data is not None for data in experience):
            self.buffer.append(experience)
        else:
            print("Invalid experience (containing None)!")
            # raise ValueError("Invalid experience (containing None)!")

    def record(self, state, action, reward, done):
        '''
        qzl: 已弃用
        '''
        # experience: [state, action, reward, next_state]
        if self.experience[0] is None or self.experience[1] is None:
            # 如果state 和 action都不为空
            self.experience[2:] = reward, state, done
            self.store_experience(self.experience)

        self.clear_experience()
        if not done:
            self.experience[:2] = state, action

    def save(self, filename, mode='wb'):
        # todo: 完成将数据保存为本地文件,但现在写的太粗糙
        file = open(filename,mode)
        pickle.dump(self.buffer, file)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch


