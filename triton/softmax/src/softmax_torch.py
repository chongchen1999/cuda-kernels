import torch

def naive_softmax(x):
    return torch.softmax(x, dim=1)
