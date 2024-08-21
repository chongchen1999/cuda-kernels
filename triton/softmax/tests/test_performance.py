import triton.testing
from benchmarks.softmax_performance import benchmark

def test_performance():
    benchmark.run(show_plots=True, print_data=True)
