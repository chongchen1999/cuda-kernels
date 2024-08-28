#include <cstdio>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include "includes/sgemm.cuh"

static const int iterations = 1000;

void initMatrix(float *A, int size) {
    std::for_each(
        A, A + size,
        [](float &a) {
            a = static_cast<float>(std::rand()) / RAND_MAX;
        }
    );
}

void launchCublasKernel(float *device_result, float *device_A, float *device_B, int M, int K, int N) {
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
            device_result, N
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
}



void launchSgemmKernel(float *device_C, float *device_A, float *device_B, int M, int K, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int block_tile_m = 128;
    int block_tile_n = 128; 
    int block_tile_k = 8;
    int thread_tile_m = 8; 
    int thread_tile_n = 8;

    int blocks_per_row = (N + block_tile_n - 1) / block_tile_n;
    int blocks_per_col = (M + block_tile_m - 1) / block_tile_m;
    dim3 grid_shape(blocks_per_row, blocks_per_col);
    dim3 block_shape(256);

    float alpha = 1.0;
    float beta = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sgemmKernel<<<grid_shape, block_shape>>>(
            device_C, device_A, device_B, 
            M, N, K, 
            alpha, beta
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

bool checkResult(float *A, float *B, int size) {
    const float eps = 1e-3;
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) / ((fabs(B[i])) + eps) > eps) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    std::srand(233u);
    int M = 512;
    int K = 512;
    int N = 512;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    float *host_A, *host_B, *host_C, *host_cublas_result;
    float *device_A, *device_B, *device_C, *device_cublas_result;

    host_A = static_cast<float *>(malloc(M * K * sizeof(float)));
    host_B = static_cast<float *>(malloc(K * N * sizeof(float)));
    host_C = static_cast<float *>(malloc(M * N * sizeof(float)));
    host_cublas_result = static_cast<float *>(malloc(M * N * sizeof(float)));

    cudaMalloc(reinterpret_cast<void **>(&device_A), M * K * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_B), K * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_C), M * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_cublas_result), M * N * sizeof(float));

    initMatrix(host_A, M * K);
    initMatrix(host_B, K * N);
    std::memset(host_C, 0, M * N * sizeof(float));
    std::memset(host_cublas_result, 0, M * N * sizeof(float));

    launchCublasKernel(device_cublas_result, device_A, device_B, M, K, N);
    cudaMemcpy(host_cublas_result, device_cublas_result, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    launchSgemmKernel(device_C, device_A, device_B, M, K, N);
    cudaMemcpy(host_C, device_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (checkResult(host_C, host_cublas_result, M * N)) {
        puts("Success!");
    }

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_cublas_result);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    cudaFree(device_cublas_result);

    return 0;
}