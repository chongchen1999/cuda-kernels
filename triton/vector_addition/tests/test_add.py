import torch
from src.add_kernel import add

def test_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    output_triton = add(x, y)
    output_torch = x + y
    
    assert torch.allclose(output_torch, output_triton, atol=1e-6), \
        "Triton and Torch outputs do not match!"
