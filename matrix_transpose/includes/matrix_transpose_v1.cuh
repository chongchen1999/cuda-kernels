#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <algorithm>
#include <cstdio>

namespace matrixTransposeV1 {
    template <typename T, int tile_size>
    __global__ void matrixTranspose(T *in, T *out, int M, int N) {
        int y = blockIdx.y * tile_size + threadIdx.y;
        int x = blockIdx.x * tile_size + threadIdx.x;
        if (x < N && y < M) {
            out[x * M + y] = in[y * N + x];
        }
    }

    template <typename T>
    void launchTranspose(T *A, T *AT, int M, int N, int iters = 1) {
        const int tile_size = 16;
        const int tiled_M = (M + tile_size - 1) / tile_size;
        const int tiled_N = (N + tile_size - 1) / tile_size;
        dim3 block_dim(tile_size, tile_size);
        dim3 grid_dim(tiled_N, tiled_M);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        for (int i = 0; i < iters; ++i) {
            matrixTranspose<T, tile_size><<<grid_dim, block_dim>>>(A, AT, M, N);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("matrixTransposeV1 Time: %f ms\n", elapsedTime);
    }
}