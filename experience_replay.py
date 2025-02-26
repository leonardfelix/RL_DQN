from collections import deque
import random
import pickle
import torch
import numpy as np

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
        self.len = maxlen

    def append(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # added prioritied experience replay
    def priority_sample(self, batch_size, device):
        priorities = torch.tensor([abs(experience[3]) for experience in self.memory], device=device)
        probabilities = priorities / priorities.sum()
        sample_indices = torch.multinomial(probabilities, batch_size, replacement=True)
        sampled_experiences = [self.memory[i] for i in sample_indices.cpu().numpy()]
        return sampled_experiences

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