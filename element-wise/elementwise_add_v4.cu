#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <memory>

const int N = 1 << 25;
const int iterations = 2000;

__global__ void addKernel_v4(float *a, float *b, float *result) {
    const int vec_gid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 vec_a = *reinterpret_cast<float4 *>(a + vec_gid);
    float4 vec_b = *reinterpret_cast<float4 *>(b + vec_gid);
    *reinterpret_cast<float4 *>(result + vec_gid) = {
        vec_a.x + vec_b.x,
        vec_a.y + vec_b.y,
        vec_a.z + vec_b.z,
        vec_a.w + vec_b.w
    };
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
    auto host_a = std::make_unique<float[]>(N);
    auto host_b = std::make_unique<float[]>(N);
    auto host_result = std::make_unique<float[]>(N);
    auto cpu_result = std::make_unique<float[]>(N);

    float *device_a, *device_b, *device_result;
    cudaMalloc(reinterpret_cast<void **>(&device_a), N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_b), N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_result), N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>((233LL * i * i  + 666LL * i) % 666233);
        cpu_result[i] = host_a[i] + host_b[i];
    }

    cudaMemcpy(device_a, host_a.get(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b.get(), N * sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (N + (block_size * 4) - 1) / (block_size * 4);

    dim3 vec_grid_shape(grid_size);
    dim3 vec_block_shape(block_size);

    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        addKernel_v4<<<vec_grid_shape, vec_block_shape>>>(device_a, device_b, device_result);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_result.get(), device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (!checkResult(host_result.get(), cpu_result.get())) {
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

    return 0;
}