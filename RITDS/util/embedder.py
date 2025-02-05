from torch import nn
import torch 
from einops import rearrange

class RTIDS_Embedder(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(mask, dim))
        self.biases = nn.Parameter(torch.randn(mask, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases