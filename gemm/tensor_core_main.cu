#include "includes/gemm_tensor_core.cuh"
#include "includes/gemm_cublas.cuh"

int main() {
    float *hostA, *hostB, *hostC, *cublas_C;
    int M = 4096;
    int K = 4096;
    int N = 4096;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    cublas_C = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++) {
        hostB[i] = i % 3;
    }

    // hostMatrix(hostA, hostB, hostC, M, K, N);
    gemm_tensor_core::hostMatrixV2(hostA, hostB, hostC, M, K, N);
    gemm_cublas::sgemm_cublas(hostA, hostB, cublas_C, M, K, N);
    
    free(hostA);
    free(hostB);
    free(hostC);
    free(cublas_C);
    return 0;
}