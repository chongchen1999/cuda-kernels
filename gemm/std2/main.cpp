#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                      \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << " \""                      \
                      << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1);                                                     \
        }                                                                \
    } while (0)

// Reference matrix multiplication on the host
void reference_matmul(float* A, float* B, float* C, int M, int N, int K,
                      float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum * alpha + beta * C[i * N + j];
        }
    }
}

// Optimized GEMM function
void OptsGemm(int m, int n, int k, float* d_A, float* d_B, float* d_C,
              float alpha, float beta);

int main() {
    // Define problem sizes and benchmark parameters
    int problem[5] = {512, 1024, 2048, 4096, 8192};
    int benchmark_times = 10;
    float alpha = 1.0f;
    float beta = 0.0f;

    for (auto matmul_size : problem) {
        float *h_A, *h_B, *h_C, *ref_C, *cublas_h_C;
        int N = matmul_size;
        int M = matmul_size;
        int K = matmul_size;
        int bytes_a = M * K * sizeof(float);
        int bytes_b = K * N * sizeof(float);
        int bytes_c = M * N * sizeof(float);

        // Allocate host memory
        h_A = (float*)malloc(bytes_a);
        h_B = (float*)malloc(bytes_b);
        h_C = (float*)malloc(bytes_c);
        cublas_h_C = (float*)malloc(bytes_c);
        ref_C = (float*)malloc(bytes_c);

        // Initialize matrices A and B with random values
        for (int i = 0; i < M * K; ++i) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        for (int i = 0; i < N * K; ++i) {
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        memset(h_C, 0, bytes_c);
        memset(cublas_h_C, 0, bytes_c);

        // Allocate device memory
        float *d_A, *d_B, *d_C, *cublas_C;
        CHECK_CUDA(cudaMalloc((void**)&d_A, bytes_a));
        CHECK_CUDA(cudaMalloc((void**)&d_B, bytes_b));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes_c));
        CHECK_CUDA(cudaMalloc((void**)&cublas_C, bytes_c));

        // Copy input matrices from host to device
        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_c, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(cublas_C, cublas_h_C, bytes_c, cudaMemcpyHostToDevice));

        // Define CUDA kernel execution configuration
        constexpr int block_len = 128;
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((M + 128 - 1) / 128, (N + 128 - 1) / 128);

        // Time the optimized implementation
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, NULL));
        for (int i = 0; i < benchmark_times; ++i) {
            OptsGemm(M, N, K, d_A, d_B, d_C, alpha, beta);
        }
        CHECK_CUDA(cudaEventRecord(stop, NULL));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = milliseconds / benchmark_times;
        double flopsPerMatrixMul = 2.0 * M * K * N;
        double gflops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0);

        std::cout << "Opt Gemm Problem size: " << M << "x" << N << "x" << K << std::endl;
        std::cout << "Used time: " << msecPerMatrixMul << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost));

        // CUBLAS compute
        cublasHandle_t handle;
        cublasCreate(&handle);

        CHECK_CUDA(cudaEventRecord(start, NULL));
        for (int i = 0; i < benchmark_times; ++i) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                        N, d_A, K, &beta, cublas_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop, NULL));
        CHECK_CUDA(cudaEventSynchronize(stop));
        milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        msecPerMatrixMul = milliseconds / benchmark_times;
        gflops = (flopsPerMatrixMul * 1.0e-9) / (msecPerMatrixMul / 1000.0);

        std::cout << "Cublas Gemm Problem size: " << M << "x" << N << "x" << K << std::endl;
        std::cout << "Used time: " << msecPerMatrixMul << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        CHECK_CUDA(cudaMemcpy(cublas_h_C, cublas_C, bytes_c, cudaMemcpyDeviceToHost));

        // Validate the result
        std::cout << "Validating the result of problem M: " << M << ", N: " << N << ", K: " << K << std::endl;
        bool error = false;
        for (int i = 0; i < M * N; ++i) {
            if (abs(h_C[i] - cublas_h_C[i]) > 0.001) {
                std::cerr << "Mismatch at index " << i << std::endl;
                std::cerr << "Opt: " << h_C[i] << " cublas: " << ref_C[i] << std::endl;
                error = true;
                break;
            }
        }
        if (!error) {
            std::cout << "Verification passed!!!" << std::endl << std::endl;
        } else {
            std::cout << "Verification failed!!!" << std::endl << std::endl;
        }

        // Free device and host memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(ref_C);
        free(cublas_h_C);
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(cublas_C));
    }

    return 0;
}