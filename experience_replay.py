from collections import deque
import random
import pickle
import torch

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
        self.len = maxlen
        self.priorities = deque([], maxlen=maxlen)

    def append(self, experience):
        self.memory.append(experience)
        self.priorities.append(abs(experience[3]))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save_memory_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.memory, file)

    def load_memory_from_file(self, filename):
        with open(filename, 'rb') as file:
            self.memory = pickle.load(file)

    # for testing
    def get_first(self):
        return self.memory[0]

    def __len__(self):
        return len(self.memory)