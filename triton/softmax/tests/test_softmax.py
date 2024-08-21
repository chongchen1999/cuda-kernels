import torch
from src.softmax_triton import softmax
from src.softmax_torch import naive_softmax

def test_softmax():
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_torch = naive_softmax(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
