from collections import deque
import random
import numpy as np


class Replay_Buffer:
    def __init__(self, max_buffer_size, batch_size, dflt_dtype):
        # self.max_buffer_size = 1000000
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.dflt_dtype = dflt_dtype

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample_batch(self):
        replay_buffer = np.array(random.sample(self.buffer, self.batch_size))
        arr = np.array(replay_buffer)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = arr[:, 1].astype(self.dflt_dtype).reshape(-1, 1)
        rewards_batch = arr[:, 2].astype(self.dflt_dtype).reshape(-1, 1)
        next_states_batch = np.vstack(arr[:, 3])
        done_batch = np.vstack(arr[:, 4]).astype(bool)
        return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch

class Prioritized_experience_replay:
    def __init__(self, max_buffer_size, batch_size, dflt_dtype):
        self.max_buffer_size = 1000000
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.priorites = deque(maxlen=max_buffer_size)
        self.indexes = deque(maxlen=max_buffer_size)
        self.dflt_dtype = dflt_dtype

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
        self.priorites.append(1)
        ln = len(self.buffer)
        if ln < self.max_buffer_size : self.indexes.append(ln)

    def update_priorities(self,indices,priorites):
        for indx,priority in zip(indices,priorites):
            self.priorites[indx-1] = priority+1

    def sample_batch(self):
        indices = random.choices(self.indexes,weights=self.priorites ,k = self.batch_size)

        replay_buffer = [self.buffer[indx-1] for indx in indices]
        arr = np.array(replay_buffer)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = arr[:, 1].astype(self.dflt_dtype).reshape(-1, 1)
        rewards_batch = arr[:, 2].astype(self.dflt_dtype).reshape(-1, 1)
        next_states_batch = np.vstack(arr[:, 3])
        done_batch = np.vstack(arr[:, 4]).astype(bool)
        return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch, indices