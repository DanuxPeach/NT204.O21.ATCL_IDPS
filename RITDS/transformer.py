import torch
from torch import nn
from .encoder import RTIDS_Encoder
from .decoder import RTIDS_Decoder

class RTIDS_Transformer(nn.Module):
    def __init__(self, numClass, dim, depth, heads, maskSize, dropout = 0.1):
        super().__init__()
        self.encoder = RTIDS_Encoder(dim, depth , heads, maskSize, dropout)
        self.decoder = RTIDS_Decoder(dim, depth , heads, maskSize, dropout)
        self.out = nn.Linear(maskSize*dim, numClass)
    
    def forward(self, src, mask=None):
        e_outputs = self.encoder(src, None)
        d_output = self.decoder(src, e_outputs, mask)
        d_intermediate = d_output.view(d_output.size(0), -1)
        output = self.out(d_intermediate)
        output = torch.softmax(output,dim=1)
        return output