#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

template <
    int block_tile_m = 128, 
    int block_tile_n = 128, 
    int block_tile_k = 8, 
    int thread_tile_m = 8, 
    int thread_tile_n = 8
>
__global__ void sgemmKernel(
    float *device_C, float *device_A, float *device_B, 
    int M, int N, int K, 
    float alpha, float beta
);