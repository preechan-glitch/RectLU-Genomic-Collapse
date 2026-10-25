import torch
import torch.nn as nn
from models.activations import RectLU, SmoothRectLU, ASRectLU

def compute_monte_carlo_gains(n_samples=2000000, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(n_samples)
    acts = {'ReLU': nn.ReLU(), 'LeakyReLU': nn.LeakyReLU(0.01), 'RectLU': RectLU(0.0, 2.0), 'SmoothRectLU': SmoothRectLU(0.0, 2.0), 'ASRectLU': ASRectLU(0.0, 2.0), 'ELU': nn.ELU(), 'SELU': nn.SELU(), 'GELU': nn.GELU(), 'SiLU': nn.SiLU(), 'Mish': nn.Mish(), 'Softplus': nn.Softplus()}
    for name, act in acts.items():
        xi = x.clone().requires_grad_(True)
        y = act(xi)
        g = torch.autograd.grad(y.sum(), xi)[0]
        print(f'{name:15s} | Gain: {(y.pow(2).mean()/xi.pow(2).mean()).item():.3f} | Bwd: {g.pow(2).mean().item():.3f}')

if __name__ == '__main__':
    compute_monte_carlo_gains()
