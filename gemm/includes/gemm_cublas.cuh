#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace gemm_cublas {
    void sgemm_cublas(
        float *host_A, float *host_B, float *hots_C, 
        int M, int K, int N,
        const int iterations = 1000
    ) {
        float *device_A, *device_B, *device_cublas;
        cudaMalloc(reinterpret_cast<void **>(&device_A), M * K * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&device_B), N * K * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&device_cublas), M * N * sizeof(float));
        cudaMemcpy(device_A, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

        float alpha = 1.0;
        float beta = 0.0;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        cublasHandle_t handle;
        cublasCreate(&handle);
        for (int i = 0; i < iterations; ++i) {
            cublasSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, 
                device_B, N, 
                device_A, K, 
                &beta, 
                device_cublas, N
            );
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float millionseconds = 0.0;
        cudaEventElapsedTime(&millionseconds, start, stop);

        millionseconds /= iterations;
        float gflops = (2.0 * M * N * K) / (1 << 30) / (millionseconds / 1000.0);
        printf("cublas used time: %f ms\n", millionseconds);
        printf("cublas performance: %f GFLOPS\n", gflops);

        cudaMemcpy(hots_C, device_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_cublas);
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}