import torch
import triton
import triton.testing
from matmul.matmul import matmul

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  
        x_vals=[128 * i for i in range(2, 20)],  
        line_arg='provider',  
        line_vals=['cublas', 'triton'],  
        line_names=["cuBLAS", "Triton"],  
        styles=[('green', '-'), ('blue', '-')],  
        ylabel="TFLOPS",  
        plot_name="matmul-performance",  
        args={},  
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
