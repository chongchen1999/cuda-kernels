#include <iostream>
#include <cuda_runtime.h>

// Kernel function
__global__ void MyKernel(int *d, int *a, int *b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

int main() {
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 64;   // Threads per block
    int numElements = 1 << 25; // Total number of elements

    // Host arrays
    int *h_a, *h_b, *h_d;

    // Device arrays
    int *d_a, *d_b, *d_d;

    // Allocate memory on host
    h_a = (int*)malloc(numElements * sizeof(int));
    h_b = (int*)malloc(numElements * sizeof(int));
    h_d = (int*)malloc(numElements * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_a, numElements * sizeof(int));
    cudaMalloc((void**)&d_b, numElements * sizeof(int));
    cudaMalloc((void**)&d_d, numElements * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numElements * sizeof(int), cudaMemcpyHostToDevice);

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0
    );

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    // Determine the number of blocks based on the number of elements and block size
    int numBlocksTotal = (numElements + blockSize - 1) / blockSize;

    // Launch the kernel
    MyKernel<<<numBlocksTotal, blockSize>>>(d_d, d_a, d_b);

    // Copy result back to host
    cudaMemcpy(h_d, d_d, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a sample of the results
    std::cout << "Sample output: " << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "d[" << i << "] = " << h_d[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_d);
    
    return 0;
}
