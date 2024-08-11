#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    int n = 1024; // Size of the vectors
    float alpha = 1.0f;
    size_t size = n * sizeof(float);

    // Host vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);

    // Initialize vectors
    for (int i = 0; i < n; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device vectors
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform vector addition: B = alpha * A + B
    cublasSaxpy(handle, n, &alpha, d_A, 1, d_B, 1);

    // Copy result back to host
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Print result (optional)
    for (int i = 0; i < 5; ++i) {
        std::cout << h_B[i] << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
    free(h_A);
    free(h_B);

    return 0;
}
