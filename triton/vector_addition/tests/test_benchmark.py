from src.benchmark import benchmark

def test_benchmark():
    sizes = [2**i for i in range(12, 15)]
    for size in sizes:
        result_triton = benchmark(size=size, provider='triton')
        result_torch = benchmark(size=size, provider='torch')
        
        assert result_triton is not None
        assert result_torch is not None
