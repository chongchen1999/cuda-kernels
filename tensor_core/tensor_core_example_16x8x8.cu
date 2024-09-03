#include "stdio.h"
#include "stdint.h"
#include "cuda_fp16.h"

template<typename T>
__global__ void set_value(T* x, int32_t elem_cnt){
    for(int i = 0; i < elem_cnt; i++){
        x[i] = static_cast<T>(i % 8); 
    }
}

__global__ void tensor_core_example_16x8x8(
    float *D, 
    uint32_t const *A, 
    uint32_t const *B, 
    float const *C
) {
    // Compute the coordinates of accesses to A and B matrices
    int ab_row = threadIdx.x / 4; // m or n dimension
    int ab_col = threadIdx.x % 4; // k dimension
    // Compute the coordinates for the accumulator matrices
    int c_row = threadIdx.x / 4;
    int c_col = 2 * (threadIdx.x % 4);
    // Compute linear offsets into each matrix
    int ab_idx = ab_row * 4 + ab_col;
    int cd_idx = c_row * 8 + c_col;

    // Issue Tensor Core operation
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[cd_idx]), "=f"(D[cd_idx+1]), "=f"(D[cd_idx+64]), "=f"(D[cd_idx+1+64])
      : 
        "r"(A[ab_idx]), "r"(A[ab_idx+32]), 
        "r"(B[ab_idx]), 
        "f"(C[cd_idx]), "f"(C[cd_idx+1]), "f"(C[cd_idx+64]), "f"(C[cd_idx+1+64])
    );
}

__global__ void printMatrix(float* result, const int m, const int n){
    for(int row = 0; row < m; row++){
        for(int col = 0; col < n; col++){
            printf("%f ", static_cast<float>(result[row * n + col])); 
        }
        printf("\n"); 
    }
}

int main(){
    half* a; 
    half* b; 
    float* c; 
    float* d; 

    const int32_t m = 16; 
    const int32_t k = 8; 
    const int32_t n = 8; 

    cudaMalloc(&a, m * k * sizeof(half)); 
    cudaMalloc(&b, k * n * sizeof(half)); 
    cudaMalloc(&c, m * n * sizeof(float)); 
    cudaMalloc(&d, m * n * sizeof(float)); 

    set_value<half><<<1, 1>>>(a, m * k); 
    set_value<half><<<1, 1>>>(b, k * n); 
    cudaMemset(c, 0, sizeof(float) * m * n); 
    cudaMemset(d, 0, sizeof(float) * m * n); 

    tensor_core_example_16x8x8<<<1, 32>>>(
        reinterpret_cast<float*>(d), 
        reinterpret_cast<uint32_t*>(a), 
        reinterpret_cast<uint32_t*>(b), 
        reinterpret_cast<float*>(c)
    );

    printMatrix<<<1, 1>>>(d, m, n); 
    cudaDeviceSynchronize(); 
    cudaFree(a); 
    cudaFree(b); 
    cudaFree(c); 
    cudaFree(d); 
    return 0;
}