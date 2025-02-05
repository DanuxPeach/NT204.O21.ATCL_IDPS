import torch

def get_mask(batchSize, numHead, size):
    mask_prob = 0.2
    mask = torch.rand((batchSize, numHead, size, size)) > mask_prob
    return mask