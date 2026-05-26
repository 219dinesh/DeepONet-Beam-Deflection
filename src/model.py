import torch
import torch.nn as nn

# ==========================================
# 1. Architecture Skeleton
# ==========================================
class BranchNet(nn.Module):
    def __init__(self, num_sensors, p_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_sensors, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, p_features)
        )
    def forward(self, u): return self.net(u)

class TrunkNet(nn.Module):
    def __init__(self, p_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, p_features)
        )
    def forward(self, y): return self.net(y)

class DeepONet(nn.Module):
    def __init__(self, num_sensors, p_features):
        super().__init__()
        self.branch = BranchNet(num_sensors, p_features)
        self.trunk = TrunkNet(p_features)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, u, y):
        b = self.branch(u) 
        t = self.trunk(y)  
        raw_output = torch.sum(b * t, dim=1, keepdim=True) + self.bias
        # The Ansatz (Boundary Condition Fix)
        return y * (1.0 - y) * raw_output

