#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include <cublas_v2.h>
#include "includes/commons.cuh"

inline const char *cublas_get_error(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED -- The cuBLAS library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED -- Resource allocation failed inside the cuBLAS library.";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE -- An unsupported value or parameter was passed to the function.";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH -- The function requires a feature absent from the device architecture.";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR -- An access to GPU memory space failed.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED -- The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR -- An internal cuBLAS operation failed.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED -- The functionality requested is not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR -- An error was detected when checking the current licensing.";
    default:
        return "CUBLAS_ERROR -- <unknown>";
    }
}

inline bool cublas_is_error(cublasStatus_t status) {
    return status != CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t gemm(
    cublasHandle_t handle,
    cublasOperation_t transA, cublasOperation_t transB,
    int m, int n, int k,
    const float* alpha,
    const half* A, int ldA,
    const half* B, int ldB,
    const float* beta,
    half* C, int ldC
) {
    return cublasGemmEx(
        handle, transA, transB,
        m, n, k,
        reinterpret_cast<const float*>(alpha),
        reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,
        reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,
        reinterpret_cast<const float*>(beta),
        reinterpret_cast<__half*>(C), CUDA_R_16F, ldC,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

void cublasMatmul(half *hC, half *hA, half *hB, int M, int N, int K) {
    half *dA, *dB, *dC;
    float alpha = 1.0f, beta = 0.0f;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    CUDA_CHECK(cudaMemcpy(hC, dC, M * N * 2, cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}