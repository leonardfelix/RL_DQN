import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_DQN=False):
        super(DQN, self).__init__()

        self.enable_dueling_DQN = enable_dueling_DQN

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_DQN:
            # Value network
            self.fc_value = nn.Linear(hidden_dim, hidden_dim//2)
            self.value = nn.Linear(hidden_dim//2, 1)

            # Advantage network
            self.fc_advantage = nn.Linear(hidden_dim, hidden_dim//2)
            self.advantage = nn.Linear(hidden_dim//2, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        if self.enable_dueling_DQN:
            v = F.relu(self.fc_value(x))
            value = self.value(v)

            a = F.relu(self.fc_advantage(x))
            advantage = self.advantage(a)

            Q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        else:
            Q = self.output(x)

        return Q 

if __name__ == "__main__":
    state_dim = 8
    action_dim = 2
    model = DQN(state_dim, action_dim)
    state = torch.rand(10, state_dim)
    output = model(state)
    print(output)
