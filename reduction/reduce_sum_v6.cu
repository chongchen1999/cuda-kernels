#include <bits/stdc++.h>
#include <cuda_runtime.h>

const int N = 1 << 25; // 2^25 elements
const int iterations = 1000;
const int warp_size = 32;

template<int block_size>
__device__ void warp_reduce_sum(float &sum) {
    if (block_size >= 32) {
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    }
    if (block_size >= 16) {
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    }
    if (block_size >= 8) {
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    }
    if (block_size >= 4) {
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    }
    if (block_size >= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
}

template <int block_size>
__global__ void sum_kernel(float *data, float *partial_sums) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;

    float sum = 0;
    for (int i = tid; i < N; i += offset) {
        sum += data[i];
    }

    __shared__ float warp_level_sums[block_size / warp_size];
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    warp_reduce_sum<block_size>(sum);
    if (lane_id == 0) {
        warp_level_sums[warp_id] = sum;
    }
    __syncthreads();

    sum = threadIdx.x < block_size / warp_size ? warp_level_sums[threadIdx.x] : 0;
    if (warp_id == 0) {
        warp_reduce_sum<block_size / warp_size>(sum);
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

void get_sum(const float *data, const int &N, float &sum) {
    for (int i = 0; i < N; ++i) {
        sum += data[i];
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    float *host_data = (float *)malloc(N * sizeof(float));
    float cpu_sum = 0;
    for (int i = 0; i < N; ++i) {
        float random_float = static_cast<float>(std::rand()) / RAND_MAX * 57.0f;
        host_data[i] = random_float;
        cpu_sum += random_float;
    }
    printf("CPU sum: %f\n", cpu_sum);

    float *device_data;
    cudaMalloc(&device_data, N * sizeof(float));

    constexpr int grid_size = 1024;
    constexpr int block_size = 1024;

    dim3 block(block_size);
    dim3 grid(grid_size);

    float *host_partial_sums = (float *)malloc(grid_size * sizeof(float));
    float *device_partial_sums;
    cudaMalloc(&device_partial_sums, grid_size * sizeof(float));

    cudaMemcpy(device_data, host_data, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        sum_kernel<block_size><<<grid, block>>>(device_data, device_partial_sums);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds / iterations);

    cudaMemcpy(host_partial_sums, device_partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_sum = 0;
    get_sum(host_partial_sums, grid_size, gpu_sum);
    if (std::abs(cpu_sum - gpu_sum) > 1e-5) {
        printf("Error: %f != %f\n", cpu_sum, gpu_sum);
    } else {
        printf("Success!\n");
    }

    // Calculate Bandwidth
    float total_data_transferred = N * sizeof(float) + grid_size * sizeof(float); // in bytes
    float average_time_per_iteration = milliseconds / iterations / 1000; // in seconds
    float bandwidth = total_data_transferred / average_time_per_iteration / (1 << 30); // in GB/s

    printf("Bandwidth: %f GB/s\n", bandwidth);

    cudaFree(device_data);
    cudaFree(device_partial_sums);
    free(host_data);
    free(host_partial_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}