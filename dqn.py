import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

if __name__ == "__main__":
    state_dim = 8
    action_dim = 2
    model = DQN(state_dim, action_dim)
    state = torch.rand(10, state_dim)
    output = model(state)
    print(output)
