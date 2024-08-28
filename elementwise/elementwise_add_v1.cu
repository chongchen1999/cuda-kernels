#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int N = 1 << 25;
const int iterations = 2000;

__global__ void add_v2(float *a, float *b, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    result[tid] = a[tid] + b[tid];
}

bool checkResult(float *a, float *b) {
    for (int i = 0; i < N; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    float *host_a = (float *) malloc(N * sizeof(float));
    float *host_b = (float *) malloc(N * sizeof(float));
    float *host_result = (float *) malloc(N * sizeof(float));
    float *cpu_result = (float *) malloc(N * sizeof(float));

    float *device_a;
    float *device_b;
    float *device_result;

    cudaMalloc((void **) &device_a, N * sizeof(float));
    cudaMalloc((void **) &device_b, N * sizeof(float));
    cudaMalloc((void **) &device_result, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        host_a[i] = i;
        host_b[i] = (N % (i + 1)) * 1.0f;
        cpu_result[i] = host_a[i] + host_b[i];
    }

    cudaMemcpy(device_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, N * sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (N - 1) / block_size + 1;

    dim3 Grid(grid_size);
    dim3 Block(block_size);

    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        add_v2<<<Grid, Block>>>(device_a, device_b, device_result);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_result, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (!checkResult(host_result, cpu_result)) {
        printf("Wrong Answer!\n");
    } else {
        printf("Success\n");
    }

    // Calculate memory bandwidth
    double total_data_transfer = 3 * N * sizeof(float); // 2 input arrays and 1 output array
    double bandwidth = (total_data_transfer / (1 << 30)) / (milliseconds / 1000.0 / iterations); // GB/s
    std::cout << "Elapsed time: " << milliseconds / 1000.0 << " seconds" << std::endl;
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_result);
    free(host_a);
    free(host_b);
    free(host_result);
    free(cpu_result);

    return 0;
}