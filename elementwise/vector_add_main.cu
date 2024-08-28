#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/includes/cpu_random.h"
#include "includes/vector_add_v5.cuh"
#include "includes/vector_add_v6.cuh"

static const int iters = 1000;

void cublas_vector_add(int n, float *d_A, float *d_B, float alpha) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        cublasSaxpy(handle, n, &alpha, d_A, 1, d_B, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cublas time: %.4f ms!\n", milliseconds / iters);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    float alpha = 1.0f;
    int n = 1 << 25;
    size_t size = n * sizeof(float);
    // Host vectors
    float *h_A = static_cast<float *>(malloc(size));
    float *h_B = static_cast<float *>(malloc(size));

    // Initialize vectors
    randomTools::fastRandomFill(h_A, n, 0.f, 1.f);
    randomTools::fastRandomFill(h_B, n, 0.f, 1.f);

    // Device vectors
    float *d_A_cublas, *d_B_cublas;
    cudaMalloc(reinterpret_cast<void **>(&d_A_cublas), size);
    cudaMalloc(reinterpret_cast<void **>(&d_B_cublas), size);
    cudaMemcpy(d_A_cublas, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_cublas, h_B, size, cudaMemcpyHostToDevice);

    float *d_A_mykernel, *d_B_mykernel;
    cudaMalloc(reinterpret_cast<void **>(&d_A_mykernel), size);
    cudaMalloc(reinterpret_cast<void **>(&d_B_mykernel), size);
    cudaMemcpy(d_A_mykernel, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_mykernel, h_B, size, cudaMemcpyHostToDevice);

    // Perform vector addition: B = alpha * A + B
    cublas_vector_add(n, d_A_cublas, d_B_cublas, alpha);
    my_vector_add_vecsize1::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    my_vector_add_vecsize4::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    cublas_vector_add(n, d_A_cublas, d_B_cublas, alpha);
    my_vector_add_vecsize1::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    my_vector_add_vecsize4::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    cublas_vector_add(n, d_A_cublas, d_B_cublas, alpha);
    my_vector_add_vecsize1::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    my_vector_add_vecsize4::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    cublas_vector_add(n, d_A_cublas, d_B_cublas, alpha);
    my_vector_add_vecsize1::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);
    my_vector_add_vecsize4::launchVectorAdd(n, d_A_mykernel, d_B_mykernel, alpha, iters);

    // Copy result back to host
    // cudaMemcpy(h_B, d_B_cublas, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A_cublas);
    cudaFree(d_B_cublas);
    free(h_A);
    free(h_B);
    cudaFree(d_A_mykernel);
    cudaFree(d_B_mykernel);

    return 0;
}
