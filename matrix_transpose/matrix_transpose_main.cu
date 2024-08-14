#include "../utils/includes/cpu_random.h"
#include "includes/matrix_transpose_v1.cuh"
#include "includes/matrix_transpose_v2.cuh"
#include "includes/matrix_transpose_v3.cuh"
#include "includes/matrix_transpose_v4.cuh"
#include <iostream>

static const int M = 4096;
static const int N = 5000;
static const int iters = 1000;

void printMatrix(float *A, int M, int N) {
    puts("");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f ", A[i * N + j]);
        }
        puts("");
    }
    puts("");
}

bool checkResult(float *A, float *B, int size) {
    for (int i = 0; i < size; ++i) {
        if (A[i] - B[i] > 1e-6) {
            return false;
        }
    }
    return true;
}

int main() {
    auto host_A = std::make_unique<float[]>(N * M);
    auto host_AT_cpu = std::make_unique<float[]>(N * M);
    auto host_AT1 = std::make_unique<float[]>(N * M);
    auto host_AT2 = std::make_unique<float[]>(N * M);
    auto host_AT3 = std::make_unique<float[]>(N * M);
    auto host_AT4 = std::make_unique<float[]>(N * M);
    randomTools::fastRandomFill<float>(host_A.get(), N * M, 0.f, 1.f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            host_AT_cpu[j * M + i] = host_A[i * N + j];
        }
    }

    float *device_A, *device_AT;
    cudaMalloc(reinterpret_cast<void **>(&device_A), N * M * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_AT), N * M * sizeof(float));
    cudaMemcpy(device_A, host_A.get(), N * M * sizeof(float), cudaMemcpyHostToDevice);

    matrixTransposeV1::launchTranspose<float>(device_A, device_AT, M, N, iters);
    cudaMemcpy(host_AT1.get(), device_AT, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    matrixTransposeV2::launchTranspose<float>(device_A, device_AT, M, N, iters);
    cudaMemcpy(host_AT2.get(), device_AT, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    matrixTransposeV3::launchTranspose<float>(device_A, device_AT, M, N, iters);
    cudaMemcpy(host_AT3.get(), device_AT, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    matrixTransposeV4::launchTranspose<float>(device_A, device_AT, M, N, iters);
    cudaMemcpy(host_AT4.get(), device_AT, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    puts("");
    matrixTransposeV1::launchTranspose<float>(device_A, device_AT, M, N, iters);
    matrixTransposeV2::launchTranspose<float>(device_A, device_AT, M, N, iters);
    matrixTransposeV3::launchTranspose<float>(device_A, device_AT, M, N, iters);
    matrixTransposeV4::launchTranspose<float>(device_A, device_AT, M, N, iters);

    std::cout << checkResult(host_AT_cpu.get(), host_AT1.get(), N * M) << std::endl;
    std::cout << checkResult(host_AT_cpu.get(), host_AT2.get(), N * M) << std::endl;
    std::cout << checkResult(host_AT_cpu.get(), host_AT3.get(), N * M) << std::endl;
    std::cout << checkResult(host_AT_cpu.get(), host_AT4.get(), N * M) << std::endl;

    cudaFree(device_A);
    cudaFree(device_AT);
    return 0;
}