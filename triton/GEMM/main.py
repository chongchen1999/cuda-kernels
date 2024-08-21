import argparse
from benchmarks.matmul_benchmark import benchmark
from tests.test_matmul import test_matmul

def run_benchmarks():
    print("Running benchmarks...")
    benchmark.run(show_plots=True, print_data=True)

def run_tests():
    print("Running tests...")
    test_matmul()
    print("All tests passed!")

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks and tests for matrix multiplication.")
    parser.add_argument('--bench', action='store_true', help="Run benchmarks")
    parser.add_argument('--test', action='store_true', help="Run tests")

    args = parser.parse_args()

    if args.bench:
        run_benchmarks()
    if args.test:
        run_tests()

    if not args.bench and not args.test:
        print("No arguments provided. Use --bench to run benchmarks and --test to run tests.")
        run_tests()
        run_benchmarks()

if __name__ == "__main__":
    main()
