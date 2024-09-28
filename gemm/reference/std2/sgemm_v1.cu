#include <iostream>
#include <cstdlib>
#include <ctime>

const int S = 1 << 10;
const int seq_len = S;
const int K = S;
const int M = S;

// CUDA kernel for SGEMM
__global__ void sgemm_kernel(float* A, float* B, float* C, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * N + k] * B[k * K + col];
        }
        C[row * N + col] = value;
    }
}

// CPU implementation for SGEMM
void sgemm_cpu(float* A, float* B, float* C, int N, int K, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            float value = 0.0f;
            for (int k = 0; k < N; ++k) {
                value += A[i * N + k] * B[k * K + j];
            }
            C[i * N + j] = value;
        }
    }
}

// Function to initialize matrices with random values
void initialize_matrix(float* matrix, const int &N, const int &M) {
    for (int i = 0; i < N * M; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to check the correctness of the CUDA result
bool check_result(const float* C_cpu, const float* C_gpu, const int &N, const int &M) {
    for (int i = 0; i < N * M; ++i) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}

void matrix_print(const float* matrix, const int &N, const int &M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    srand(time(nullptr));

    // Allocate host memory
    float* host_A = (float*)malloc(seq_len * K * sizeof(float));
    float* host_B = (float*)malloc(K * M * sizeof(float));
    float* host_C_cpu = (float*)malloc(seq_len * M * sizeof(float));
    float* host_C_gpu = (float*)malloc(seq_len * M * sizeof(float));

    // Initialize matrices
    initialize_matrix(host_A, seq_len, K);
    initialize_matrix(host_B, K, M);

    // Allocate device memory
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, seq_len * K * sizeof(float));
    cudaMalloc(&device_B, K * M * sizeof(float));
    cudaMalloc(&device_C, seq_len * M * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(device_A, host_A, seq_len * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 Block(16, 16);
    dim3 Grid((seq_len + Block.x - 1) / Block.x, (seq_len + Block.y - 1) / Block.y);

    // Launch the kernel
    sgemm_kernel<<<Grid, Block>>> (device_A, device_B, device_C, seq_len, K, M);

    // Copy result back to host
    cudaMemcpy(host_C_gpu, device_C, seq_len * M * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform SGEMM on CPU
    sgemm_cpu(host_A, host_B, host_C_cpu, seq_len, K, M);

    // Check results
    if (check_result(host_C_cpu, host_C_gpu, seq_len, M)) {
        std::cout << "SGEMM implementation is correct!" << std::endl;
    } else {
        std::cout << "SGEMM implementation is incorrect!" << std::endl;
        matrix_print(host_C_cpu, seq_len, M);
        printf("\n");
        matrix_print(host_C_gpu, seq_len, M);
    }

    // Free memory
    free(host_A);
    free(host_B);
    free(host_C_cpu);
    free(host_C_gpu);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
}