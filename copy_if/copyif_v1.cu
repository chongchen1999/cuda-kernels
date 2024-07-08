#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// Kernel for predicate evaluation
template <typename T, typename Predicate>
__global__ void evaluate_predicate(const T* input, int* flags, int n, Predicate pred) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        flags[idx] = pred(input[idx]) ? 1 : 0;
    }
}

// Kernel for copying elements based on the flags and positions
template <typename T>
__global__ void copy_if_kernel(const T* input, T* output, const int* positions, const int* flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx]) {
        output[positions[idx]] = input[idx];
    }
}

// Host function
template <typename T, typename Predicate>
void copy_if(const thrust::host_vector<T>& h_input, thrust::host_vector<T>& h_output, Predicate pred) {
    int n = h_input.size();
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    thrust::device_vector<T> d_input = h_input;
    thrust::device_vector<int> d_flags(n);
    thrust::device_vector<int> d_positions(n);
    thrust::device_vector<T> d_output(n);

    // Evaluate predicate
    evaluate_predicate<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_input.data()),
                                                 thrust::raw_pointer_cast(d_flags.data()),
                                                 n, pred);
    cudaDeviceSynchronize();

    // Compute positions using exclusive scan
    thrust::exclusive_scan(d_flags.begin(), d_flags.end(), d_positions.begin());

    // Copy elements based on positions
    copy_if_kernel<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_input.data()),
                                             thrust::raw_pointer_cast(d_output.data()),
                                             thrust::raw_pointer_cast(d_positions.data()),
                                             thrust::raw_pointer_cast(d_flags.data()),
                                             n);
    cudaDeviceSynchronize();

    // Get the number of selected elements
    int new_size = thrust::reduce(d_flags.begin(), d_flags.end());

    h_output.resize(new_size);
    thrust::copy(d_output.begin(), d_output.begin() + new_size, h_output.begin());
}

// Usage example
int main() {
    thrust::host_vector<int> h_input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    thrust::host_vector<int> h_output;

    auto pred = [] __device__ (int x) { return x % 2 == 0; };

    copy_if(h_input, h_output, pred);

    std::cout << "Filtered elements: ";
    for (int x : h_output) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
