#include <cstdio>
#include <cstdint>

__global__ void set_value(int8_t* x, int32_t elem_cnt) {
    for(int i = 0; i < elem_cnt; ++i) {
        x[i] = static_cast<int8_t>(i % 8); 
    }
}

// Do AxB + C = D. 
__global__ void tensor_core_example_8x8x16(
    int32_t *D, 
    uint32_t const *A, 
    uint32_t const *B, 
    int32_t const *C
) {
    // Compute the coordinates of accesses to A and B matrices
    int outer = threadIdx.x / 4; // m or n dimension
    int inner = threadIdx.x % 4; // k dimension

    // Compute the coordinates for the accumulator matrices
    int c_row = threadIdx.x / 4;
    int c_col = 2 * (threadIdx.x % 4);

    // Compute linear offsets into each matrix
    int ab_idx = outer * 4 + inner;
    int cd_idx = c_row * 8 + c_col;
    
    // Issue Tensor Core operation
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=r"(D[cd_idx]), "=r"(D[cd_idx ^ 1])
        : "r"(A[ab_idx]), "r"(B[ab_idx]), "r"(C[cd_idx]), "r"(C[cd_idx+1])
    );
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