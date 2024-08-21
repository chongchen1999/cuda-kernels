from src.add_kernel import add
from src.benchmark import benchmark

import torch

def main():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    output_torch = x + y
    output_triton = add(x, y)
    
    print(f'在torch和triton之间的最大差异是 '
          f'{torch.max(torch.abs(output_torch - output_triton))}')
    
    # benchmark.run(show_plots=True)

if __name__ == "__main__":
    main()
