import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPONetSimple(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PPONetSimple, self).__init__()
        self.state_linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),  # size = (B, 32)
        )
        self.action_logits = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.state_linear(x)
        value = self.value(x)
        value = torch.squeeze(value, dim=-1)
        logits = self.action_logits(x)
        return logits, value



class PPONetSimple2(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PPONetSimple2, self).__init__()
        self.state_linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),  # size = (B, 32)
        )
        self.classifier = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.state_linear(x)
        value = self.value(x)
        value = torch.squeeze(value, dim=-1)
        logits = self.classifier(x)
        return logits, value



class PPONetDynamic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim1, action_dim2):
        super(PPONetDynamic, self).__init__()
        self.state_linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),  # size = (B, 32)
        )
        self.action_logits = nn.Linear(hidden_dim, action_dim1)
        self.action_logits2 = nn.Linear(hidden_dim, action_dim2)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.state_linear(x)
        value = self.value(x)
        value = torch.squeeze(value, dim=-1)
        logits1 = self.action_logits(x) # k index (action) distribution
        logits2 = self.action_logits2(x) # length
        return logits1, logits2, value



class PPONetDynamic2(nn.Module):
    # multi-discrete action space
    def __init__(self, state_dim, hidden_dim, action_dim1, action_dim2):
        super(PPONetDynamic2, self).__init__()
        self.state_linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),  # size = (B, 32)
        )
        self.action_logits1 = nn.Linear(hidden_dim, action_dim1)
        self.action_logits2 = nn.Linear(hidden_dim, action_dim2)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.state_linear(x)
        value = self.value(x)
        value = torch.squeeze(value, dim=-1)
        logits1 = self.action_logits1(x)
        logits2 = self.action_logits2(x)
        return logits1, logits2, value
