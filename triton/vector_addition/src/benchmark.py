import time
import torch
import triton
import triton.testing

from src.add_kernel import add

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    start_time = time.time()  # 记录开始时间
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    
    end_time = time.time()  # 记录结束时间
    
    gbps = lambda ms: 12 * size / ms * 1e-6
    avg_time = end_time - start_time  # 计算运行时间
    
    print(f'Provider: {provider}, Size: {size}, Time: {avg_time:.4f} seconds')
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)
