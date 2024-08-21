import torch

def generate_random_tensor(shape):
    return torch.randn(shape, device='cuda')
