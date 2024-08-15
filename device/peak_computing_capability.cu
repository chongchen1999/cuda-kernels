#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"

static const int loop_times = 1000;

// NVIDIA RTX 3070 Laptop GPU: 11.3 TFLOPS
__global__ void FP32FLOPS(
    int *start, 
    int *stop, 
    const float *x, 
    const float *y, 
    float *result
) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    float d1 = x[gid];
    float d2 = y[gid];
    int start_time = 0;

    float res = 0.0f;

    // Only measure the computation time, eliminate the memory access time
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start_time) :: "memory");

    for (int i = 0; i < loop_times; ++i) {
        res = d1 * d2 + res;
        d1 = d2 * res + d1;
        d2 = res * d1 + d2;
        res = d1 * d2 + res;
    }

    asm volatile("bar.sync 0;");

    int stop_time = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop_time) :: "memory");

    start[gid] = start_time;
    stop[gid] = stop_time;
    result[gid] = res;
}

int main() {
    int N = 1024;

    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));

    float *d_x;
    float *d_y;

    cudaMalloc((void **)&d_x, N * sizeof(float)); 
    cudaMalloc((void **)&d_y, N * sizeof(float)); 

    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_result;
    int *startClock = (int *)malloc(N * sizeof(int));
    int *stopClock = (int *)malloc(N * sizeof(int));

    int *d_startClock;
    int *d_stopClock;

    cudaMalloc((void **)&d_result, N * sizeof(float)); 
    cudaMalloc((void **)&d_startClock, N * sizeof(int)); 
    cudaMalloc((void **)&d_stopClock, N * sizeof(int)); 

    // Confirm launch max threads of SM = 1024 to do FMA to saturate SM resource
    FP32FLOPS<<<1, N>>>(
        d_startClock, 
        d_stopClock, 
        d_x, 
        d_y, 
        d_result
    );

    cudaMemcpy(startClock, d_startClock, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock, d_stopClock, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    const int threads_per_SM = props.maxThreadsPerMultiProcessor;
    const int SM_num = props.multiProcessorCount;
    float FLOPS = (loop_times * 4 * 2 * N) / (static_cast<float>(stopClock[0] - startClock[0]));

    printf("GPU Max Clock rate: %0.2f GHz\n", props.clockRate * 1e-6f);
    printf("SM counts: %d\n", SM_num);
    printf("Threads per SM: %d\n", threads_per_SM);
    printf("Actual NVIDIA GPU peak FLOPS: %f TFLOPS\n", FLOPS * props.clockRate * 1e-9 * SM_num);

    free(x);
    free(y);
    free(startClock);
    free(stopClock);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cudaFree(d_startClock);
    cudaFree(d_stopClock);

    return 0;
}
