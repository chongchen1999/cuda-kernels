#pragma once

#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace my_vector_add_vecsize4 {
    __global__ void vectorAdd(float *a, float *b, int n, float alpha) {
        const int vec_gid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
        if (vec_gid >= n) {
            return;
        }
        float4 &vec_a = *reinterpret_cast<float4 *>(a + vec_gid);
        float4 &vec_b = *reinterpret_cast<float4 *>(b + vec_gid);
        vec_b.x += alpha * vec_a.x;
        vec_b.y += alpha * vec_a.y;
        vec_b.z += alpha * vec_a.z;
        vec_b.w += alpha * vec_a.w;
    }

    // B = alpha * A + B
    void launchVectorAdd(int n, float *a, float *b, float alpha, int iters) {
        int block_size = 256;
        int grid_size = (n + (block_size * 4) - 1) / (block_size * 4);
        dim3 grid_shape(grid_size);
        dim3 block_shape(block_size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            vectorAdd<<<grid_shape, block_shape>>>(a, b, n, alpha);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("my vectorized vector add time: %.4f ms!\n", milliseconds / iters);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}