from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np

# To store the trajectories in the memory
Transition = namedtuple('Transition', 'state, action, reward, policy')

class TrajectoryReplayMemory(object):
    def __init__(self, nb_episodes, max_episode_length):
        self.nb_episodes = nb_episodes
        self.max_episode_length = max_episode_length
        self.memory = deque(maxlen = nb_episodes)
        self.memory.append([])
        self.pos = 0

    def append(self, state, action, reward, policy):
        if action is not None:
            self.memory[self.pos].append(Transition[state, action, reward, policy])
        else:
            self.memory[self.pos].append(Transition[state, None, None, None])
            self.pos = (self.pos + 1)%self.nb_episodes

    def sample_trajectory(self, max_length=0, trajectory_num=None):
        l = len(self.memory)
        if (l>0):
            if trajectory_num is None:
                epi = random.randrange(l)
            else:
                epi = trajectory_num    
            trajectory = self.memory[epi]
            t = len(trajectory)
            if not max_length:
                p = random.randrange(t - max_length - 1)
                return trajectory[p : p + max_length + 1] 
            else:
                return trajectory
        else:
            return None
    
    def sample(self, max_length=0, batch_size=1):
        l = len(self.memory)
        batch_idx = random.sample(range(l), batch_size)
        batch = [self.sample_trajectory(max_length, i) for i in batch_idx]
        minimum_length = min(len(i) for i in batch)
        # Make trajectory of equal length
        batch = [i[:minimum_length] for i in batch]

    def __len__(self):
        sz = 0
        for trajectory in self.memory:
            sz += len(trajectory)
        return sz