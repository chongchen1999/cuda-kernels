#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace gemm_cublas {
    void sgemm_cublas_tensorcore(
        float *host_A, float *host_B, float *host_C,
        int M, int K, int N,
        const int iterations = 1000
    ) {
        float *device_A, *device_B, *device_C;
        cudaMalloc(reinterpret_cast<void **>(&device_A), M * K * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&device_B), N * K * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&device_C), M * N * sizeof(float));

        cudaMemcpy(device_A, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cublasHandle_t handle;
        cublasCreate(&handle);

        // Enable tensor cores
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

        cudaEventRecord(start, 0);

        for (int i = 0; i < iterations; ++i) {
            cublasSgemmEx(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        device_B, CUDA_R_32F, N,
                        device_A, CUDA_R_32F, K,
                        &beta,
                        device_C, CUDA_R_32F, N);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        milliseconds /= iterations;

        float gflops = (2.0f * M * N * K) / (1 << 30) / (milliseconds / 1000.0f);

        printf("cublas with Tensor Core used time: %f ms\n", milliseconds);
        printf("cublas with Tensor Core performance: %f GFLOPS\n", gflops);

        cudaMemcpy(host_C, device_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

}