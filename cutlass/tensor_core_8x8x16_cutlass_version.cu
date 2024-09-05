#include <cstdio>
#include <cstdint>
#include "cutlass/arch/mma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"

__global__ void set_value(int8_t* x, int32_t elem_cnt) {
    for(int i = 0; i < elem_cnt; ++i) {
        x[i] = static_cast<int8_t>(i % 8); 
    }
}

// Do AxB + C = D. 
__global__ void cutlass_mma_example_8x8x16(
    int32_t *D,
    uint32_t const *A,
    uint32_t const *B,
    int32_t const *C
) {
    using MmaCore = typename cutlass::arch::Mma<
        cutlass::gemm::GemmShape<8, 8, 16>,
        32,
        int8_t,
        cutlass::layout::RowMajor,
        int8_t,
        cutlass::layout::ColumnMajor,
        int32_t,
        cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAddSaturate
    >::MmaCore;

    // Compute the coordinates of accesses to A and B matrices
    int outer = threadIdx.x / 4; // m or n dimension
    int inner = threadIdx.x % 4; // k dimension

    // Compute the coordinates for the accumulator matrices
    int c_row = threadIdx.x / 4;
    int c_col = 2 * (threadIdx.x % 4);

    // Compute linear offsets into each matrix
    int ab_idx = outer * 4 + inner;
    int cd_idx = c_row * 8 + c_col;

    // Shared memory for the Mma operation
    __shared__ int8_t smem_a[MmaCore::Shape::kM * MmaCore::Shape::kK];
    __shared__ int8_t smem_b[MmaCore::Shape::kN * MmaCore::Shape::kK];

    // Load A and B from global memory to shared memory
    reinterpret_cast<int32_t*>(smem_a)[ab_idx] = A[ab_idx];
    reinterpret_cast<int32_t*>(smem_b)[ab_idx] = B[ab_idx];

    __syncthreads();

    // Initialize accumulator with C values
    int32_t accum[2];
    accum[0] = C[cd_idx];
    accum[1] = C[cd_idx + 1];

    // Instantiate the Mma
    MmaCore mma;

    // Compute matrix product
    mma(accum, smem_a, smem_b, accum);

    // Store results to D
    D[cd_idx] = accum[0];
    D[cd_idx + 1] = accum[1];
}

__global__ void printMatrix(int32_t *result, const int m, const int n){
    for(int row = 0; row < m; row++){
        for(int col = 0; col < n; col++){
            int idx = row * n + col;
            printf("%d ", result[idx]);
        }
        printf("\n");
    }
}

int main(){
    int8_t * a; 
    int8_t * b; 
    int32_t *c; 
    int32_t *d; 

    const int32_t m = 8; 
    const int32_t k = 16; 
    const int32_t n = 8; 

    cudaMalloc(&a, m * k * sizeof(int8_t)); 
    cudaMalloc(&b, k * n * sizeof(int8_t)); 
    cudaMalloc(&c, m * n * sizeof(int32_t)); 
    cudaMalloc(&d, m * n * sizeof(int32_t)); 

    set_value<<<1, 1>>>(a, m * k); 
    set_value<<<1, 1>>>(b, k * n); 

    cudaMemset(c, 0, sizeof(int32_t) * m * n); 
    cudaMemset(d, 0, sizeof(int32_t) * m * n); 

    tensor_core_example_8x8x16<<<1, 32>>>(
        reinterpret_cast<int32_t*>(d), 
        reinterpret_cast<uint32_t*>(a), 
        reinterpret_cast<uint32_t*>(b), 
        reinterpret_cast<int32_t*>(c)
    ); 

    printMatrix<<<1, 1>>>(d, m, n); 
    cudaDeviceSynchronize(); 

    cudaFree(a); 
    cudaFree(b); 
    cudaFree(c); 
    cudaFree(d); 
    return 0;
}