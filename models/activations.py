import torch
import torch.nn as nn
import torch.nn.functional as F

class RectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = L, W
    def forward(self, x):
        c1 = x < self.L
        c2 = (x >= self.L) & (x < self.L + self.W)
        return torch.where(c1, torch.zeros_like(x), torch.where(c2, x*(x-self.L)/self.W, x-self.L-0.5*self.W))

class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = L, W
    def forward(self, x):
        c1 = x < self.L
        c2 = (x >= self.L) & (x < self.L + self.W)
        return torch.where(c1, torch.zeros_like(x), torch.where(c2, ((x-self.L)**2)/(2.0*self.W), x-self.L-0.5*self.W))

class ASRectLU(nn.Module):
    def __init__(self, init_L=0.0, init_W=2.0):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(float(init_L)))
        self.raw_W = nn.Parameter(torch.log(torch.exp(torch.tensor(float(init_W))) - 1.0))
    def forward(self, x):
        W = F.softplus(self.raw_W) + 1e-6
        c1 = x < self.L
        c2 = (x >= self.L) & (x < self.L + W)
        return torch.where(c1, torch.zeros_like(x), torch.where(c2, ((x-self.L)**2)/(2.0*W), x-self.L-0.5*W))
