#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include "includes/commons.cuh"

int STAGES = 1;
int MULTI_THREADING = 1;

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16, 
    int warps_m = 2, int warps_n = 2
>
extern __global__ void matmul(
    half *A, half *B, half *C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
);

float alpha = 1.0;
float beta = 0.0;

void cpuMatmul(half *C, half *A, half *B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[j * M + k]);
            }
            C[i * N + j] = __float2half(sum);
        }
    }
}

bool checkResult(half *cpu_result, half *gpu_result, int M, int N, const float eps = 1e-3) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float abs_diff = fabs(__half2float(cpu_result[i * N + j]) - __half2float(gpu_result[i * N + j]));
            const float rel_diff = abs_diff / __half2float(cpu_result[i * N + j]);
            if (abs_diff > eps && rel_diff > eps) {
                std::cout << "Mismatch at (" << i << ", " << j << ") CPU = " 
                    << __half2float(cpu_result[i * N + j]) << ", GPU = " 
                    << __half2float(gpu_result[i * N + j]) << "\n";
                return false;
            }
        }
    }
    return true;
}

extern void cublasMatmul(half *C, half *A, half *B, int M, int N, int K);

int main(int argc, char *argv[]) {
    if (argc > 1) {
        assert((argc - 1) % 2 == 0);
        for (int i = 1; i < argc; i += 2) {
            char *key = argv[i];
            char *value = argv[i + 1];
            std::string keys(key);
            if (keys == "stages")
            {
                STAGES = std::atoi(value);
                std::cout << "Setting to " << STAGES << " stages.\n";
            }
            else if (keys == "multi_threading")
            {
                MULTI_THREADING = std::atoi(value);
                std::cout << "Setting to " << MULTI_THREADING << "x threading.\n";
            }
            else if (keys == "iters") {
                iterations = std::atoi(value);
                std::cout << "Testing iters = " << iterations << ".\n";
            }
        }
    }

    srand(666233);
    half *hA = reinterpret_cast<half *>(malloc(M * K * sizeof(half)));
    half *hB = reinterpret_cast<half *>(malloc(K * N * sizeof(half)));
    half *hC = reinterpret_cast<half *>(malloc(M * N * sizeof(half)));
    half *h_cublas = reinterpret_cast<half *>(malloc(M * N * sizeof(half)));

    for (int i = 0; i < M * K; ++i) {
        hA[i] = __float2half(randFP32());
    }

    for (int i = 0; i < K * N; ++i) {
        hB[i] = __float2half(randFP32());
    }

    for (int i = 0; i < M * N; ++i) {
        hC[i] = __float2half(.0f);
    }

    cublasMatmul(h_cublas, hA, hB, M, N, K);
    puts("cublas matmul done!\n");

    half *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dA), M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dB), K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dC), M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 dimGrid((N + block_tile_n - 1) / block_tile_n, (M + block_tile_m - 1) / block_tile_m);
    dim3 dimBlock(block_tile_k, 2 * MULTI_THREADING, 2);

    int smem_size = MAX(STAGES * 128 * 32 * 2 * sizeof(half), 128 * 128 * 4);
    
    if (smem_size >= (48 << 10)) {
        CUDA_CHECK(
            cudaFuncSetAttribute(
                matmul<128, 128, 32, 64, 64, 16, 16, 16, 16, 2, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size
            )
        );
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    for (int i = 0; i < iterations / 20 + 1; ++i) {
        matmul<128, 128, 32, 64, 64, 16, 16, 16, 16, 2, 2><<<dimGrid, dimBlock, smem_size>>>(dA, dB, dC, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        matmul<128, 128, 32, 64, 64, 16, 16, 16, 16, 2, 2><<<dimGrid, dimBlock, smem_size>>>(dA, dB, dC, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaMemcpy(hC, dC, M * N * 2, cudaMemcpyDeviceToHost));

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost of CUDA kernel is " << double(ms) / iterations << "ms\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (double(ms) / iterations) * 1e3 / 1e12 << "\n";

    if (checkResult(h_cublas, hC, M, N)) {
        puts("pass!");
    } else {
        puts("fail!");
    }

    free(hA);
    free(hB);
    free(hC);
    free(h_cublas);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}