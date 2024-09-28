#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

__global__ void matmul(
    half *A, half *B, half *C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float accum = 0.0f;
        for (int i = 0; i < K; ++i) {
            accum += __half2float(A[row * K + i]) * __half2float(B[col * K + i]);
        }
        C[row * N + col] = __float2half(alpha * accum + beta * __half2float(C[row * N + col]));
    }
}