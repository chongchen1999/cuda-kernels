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

extern __global__ void matmul(
    half *A, half *B, half *C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
);

extern void cublasMatmul(half *C, half *A, half *B, int M, int N, int K);

float alpha = 1.0;
float beta = 0.0;

void cpuMatmul(half *C, half *A, half *B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[j * K + k]);
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
                /*std::cout << "Mismatch at (" << i << ", " << j << ") CPU = " 
                    << __half2float(cpu_result[i * N + j]) << ", GPU = " 
                    << __half2float(gpu_result[i * N + j]) << "\n";*/
                return false;
            }
        }
    }
    return true;
}

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
    half *h_cpu = reinterpret_cast<half *>(malloc(M * N * sizeof(half)));

    for (int i = 0; i < M * K; ++i) {
        hA[i] = __float2half(randFP32());
    }

    for (int i = 0; i < K * N; ++i) {
        hB[i] = __float2half(randFP32());
    }

    for (int i = 0; i < M * N; ++i) {
        hC[i] = __float2half(0.0f);
    }

    // cublasMatmul(h_cublas, hA, hB, M, N, K);
    // puts("cublas matmul done!\n");

    // cpuMatmul(h_cpu, hA, hB, M, N, K);
    // puts("cpu matmul done!\n");

    half *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dA), M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dB), K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dC), M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * sizeof(half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // printf("%d %d\n", dimGrid.x, dimGrid.y);

    // warmup
    for (int i = 0; i < iterations / 20 + 1; ++i) {
        matmul<<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        matmul<<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaMemcpy(hC, dC, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost of CUDA kernel is " << double(ms) / iterations << "ms\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (double(ms) / iterations) * 1e3 / 1e12 << "\n";

    // puts(checkResult(h_cublas, h_cpu, M, N) ? "cubals and cpu match!" : "cubals and cpu not match!");
    // puts(checkResult(h_cublas, hC, M, N) ? "cubals and kernel match!" : "cubals and kernel not match!");
    // puts(checkResult(h_cpu, hC, M, N) ? "cpu and kernel match!" : "cpu and kernel not match!");

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