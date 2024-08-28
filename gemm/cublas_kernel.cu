#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void generateRandomMatrix(std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void checkResults(const std::vector<float>& CPU_result, const std::vector<float>& GPU_result, int size) {
    for (int i = 0; i < size; ++i) {
        if (abs(CPU_result[i] - GPU_result[i]) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": CPU result = " << CPU_result[i] << ", GPU result = " << GPU_result[i] << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}

int main() {
    // Matrix dimensions
    int M = 512; // Rows in A and C
    int K = 512; // Columns in A, Rows in B
    int N = 512; // Columns in B and C

    // Seed for random number generation
    std::srand(static_cast<unsigned>(std::time(0)));

    // Host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_gpu(M * N);

    // Generate random matrices
    generateRandomMatrix(h_A, M, K);
    generateRandomMatrix(h_B, K, N);

    // Perform CPU matrix multiplication
    matrixMultiplyCPU(h_A, h_B, h_C, M, N, K);

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cuBLAS
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check results
    checkResults(h_C, h_C_gpu, M * N);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
