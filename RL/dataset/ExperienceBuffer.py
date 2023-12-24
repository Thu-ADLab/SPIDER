import random
from collections import deque
import pickle


class ExperienceBuffer:
    # todo:以后替换成torch.dataset
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
        '''
        弃用
        '''
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


# import torch
# from torch.utils.data import Dataset, DataLoader
# import random
#
# class ExperienceBuffer(Dataset):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0
#
#     def add_experience(self, state, action, reward, next_state, done):
#         experience = (state, action, reward, next_state, done)
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(experience)
#         else:
#             self.buffer[self.position] = experience
#         self.position = (self.position + 1) % self.capacity
#
#     def __len__(self):
#         return len(self.buffer)
#
#     def __getitem__(self, idx):
#         return self.buffer[idx]
#
#     def sample_batch(self, batch_size):
#         return random.sample(self.buffer, min(batch_size, len(self.buffer)))
#
# # Example usage:
# # Initialize the experience buffer with a capacity of 1000
# experience_buffer = ExperienceBuffer(capacity=1000)
#
# # Add experiences to the buffer
# for _ in range(1500):
#     state, action, reward, next_state, done = ..., ..., ..., ..., ...
#     experience_buffer.add_experience(state, action, reward, next_state, done)
#
# # Sample a batch of experiences
# batch_size = 32
# sampled_batch = experience_buffer.sample_batch(batch_size)
#
# # Create a DataLoader for easy batch iteration
# dataloader = DataLoader(experience_buffer, batch_size=batch_size, shuffle=True)
#
# # Iterate over batches
# for batch in dataloader:
#     # Perform training using the sampled batch
#     # Your training code here
#     pass