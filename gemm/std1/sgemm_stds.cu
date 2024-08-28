#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define checkCudaErrors(func) \
{ \
    cudaError_t e = (func); \
    if(e != cudaSuccess) \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \
}

__device__ __forceinline__ void load_global_to_bufferReg(float * const &buffer, const float * const &A, 
    const int &bm, const int &bn, const int &M, const int &N, 
    const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        *(reinterpret_cast<float4 *>(buffer) + offset) = 
            *(reinterpret_cast<const float4 *>(A + (start_row + i) * N + start_col));
    }
}

__device__ __forceinline__ void load_bufferReg_to_shared_a(const float * const &buffer, float * const &shared, 
    const int &bm, const int &bn, const int &M, const int &N, 
    const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        const float4 temp_reg = *(reinterpret_cast<const float4 *>(buffer) + offset);
        *(shared + (start_row + i) + bm * start_col) = temp_reg.x;
        *(shared + (start_row + i) + bm * (start_col + 1)) = temp_reg.y;
        *(shared + (start_row + i) + bm * (start_col + 2)) = temp_reg.z;
        *(shared + (start_row + i) + bm * (start_col + 3)) = temp_reg.w;
    }
}

__device__ __forceinline__ void load_bufferReg_to_shared_b(const float * const &buffer, float * const &shared, 
    const int &bm, const int &bn, const int &M, const int &N, 
    const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        *(reinterpret_cast<float4 *>(shared + (start_row + i) * bn + start_col)) = 
            *(reinterpret_cast<const float4 *>(buffer) + offset);
    }
}

__device__ __forceinline__ void load_shared_to_reg(float * const &reg, const float * const &shared, 
    const int &bm, const int &bn, const int &len, 
    const int &offset_row, const int &offset_col) {
    for (int i = 0; i < len; i += 4) {
        *(reinterpret_cast<float4 *>(reg + i)) = 
            *(reinterpret_cast<const float4 *>(shared + offset_row * bn + (offset_col + i)));
    }
}

namespace gemm_computational_opt {
    #define BM block_tile_m
    #define BN block_tile_n
    #define BK block_tile_k
    #define TM thread_tile_m
    #define TN thread_tile_n
    #define a_frag shared_to_reg_a
    #define b_frag shared_to_reg_b
    #define ldg_a_reg bufferReg_global_to_shared_a
    #define ldg_b_reg bufferReg_global_to_shared_b
    #define ldg_a_num loads_per_thread_a
    #define ldg_b_num loads_per_thread_b
    #define As shared_a
    #define Bs shared_b
    #define a_tile_row global_start_row_a
    #define a_tile_col global_start_col_a
    #define a_tile_stride global_stride_a
    #define b_tile_row global_start_row_b
    #define b_tile_col global_start_col_b
    #define b_tile_stride global_stride_b
    #define tx thread_x
    #define ty thread_y
    #define write_index load_index_shared
    
    #define CEIL_DIV(M, N) ((M) + (N)-1) / (N)


    template <int block_tile_m, int block_tile_n, int block_tile_k, 
        int thread_tile_m, int thread_tile_n>
    __global__ void sgemm(float *device_A, float *device_B, float *device_C, 
        int M, int K, int N, float alpha, float beta) {
        const int tid = threadIdx.x;
        const int thread_processed_tile_x_dim = block_tile_n / thread_tile_n;
        const int thread_processed_tile_y_dim = block_tile_m / thread_tile_m;
        const int thread_num = thread_processed_tile_x_dim * thread_processed_tile_y_dim;
        const int thread_x = threadIdx.x % thread_processed_tile_x_dim * TN;
        const int thread_y = threadIdx.x / thread_processed_tile_x_dim * TM;
        const int ldg_a_num = block_tile_m * block_tile_k / (thread_num << 2);
        const int ldg_b_num = block_tile_k * block_tile_n / (thread_num << 2);

        int global_start_row_a = tid / (block_tile_k >> 2);
        int global_start_col_a = (tid % (block_tile_k >> 2)) << 2;
        int global_stride_a = block_tile_m / loads_per_thread_a;

        int global_start_row_b = tid / (block_tile_n >> 2);
        int global_start_col_b = (tid % (block_tile_n >> 2)) << 2;
        int global_stride_b = block_tile_k / loads_per_thread_b;

        __shared__ float shared_a[2][block_tile_m * block_tile_k];
        __shared__ float shared_b[2][block_tile_k * block_tile_n];
        float accum[thread_tile_m][thread_tile_n] = {0.0};

        float bufferReg_global_to_shared_a[loads_per_thread_a << 2] = {0.0};
        float bufferReg_global_to_shared_b[loads_per_thread_b << 2] = {0.0};

        float shared_to_reg_a[2][thread_tile_m];
        float shared_to_reg_b[2][thread_tile_n];

        const float *A = device_A + blockIdx.y * block_tile_m * K;
        const float *B = device_B + blockIdx.x * block_tile_n;
        float *C = device_C + blockIdx.y * block_tile_m * N + blockIdx.x * block_tile_n;

        /*load_global_to_bufferReg(bufferReg_global_to_shared_a, A, block_tile_m, block_tile_k, M, K, 
            global_start_row_a, global_start_col_a, global_stride_a);
        load_global_to_bufferReg(bufferReg_global_to_shared_b, B, block_tile_k, block_tile_n, K, N, 
            global_start_row_b, global_start_col_b, global_stride_b);

        load_bufferReg_to_shared_a(bufferReg_global_to_shared_a, shared_a[0], block_tile_m, block_tile_k, 
            M, K, global_start_row_a, global_start_col_a, global_stride_a);
        load_bufferReg_to_shared_b(bufferReg_global_to_shared_b, shared_b[0], block_tile_k, block_tile_n, 
            K, N, global_start_row_b, global_start_col_b, global_stride_b);*/

        load_global_to_bufferReg(ldg_a_reg, A, BM, BK, M, K, a_tile_row, a_tile_col, a_tile_stride);
        load_global_to_bufferReg(ldg_b_reg, B, BK, BN, K, N, b_tile_row, b_tile_col, b_tile_stride);

        load_bufferReg_to_shared_a(ldg_a_reg, As[0], BM, BK, M, K, a_tile_row, a_tile_col, a_tile_stride);
        load_bufferReg_to_shared_b(ldg_b_reg, Bs[0], BK, BN, K, N, b_tile_row, b_tile_col, b_tile_stride);

        __syncthreads();

        load_shared_to_reg(a_frag[0], As[0], BK, BM, TM, 0, ty);
        load_shared_to_reg(b_frag[0], Bs[0], BK, BN, TN, 0, tx);

        int write_index = 0;
        for (int k = 0; k < K; k += BK) {
            write_index ^= 1;
            int next_k = k + BK;
            if (next_k < K) {
                load_global_to_bufferReg(ldg_a_reg, A, BM, BK, M, K, a_tile_row, a_tile_col + next_k, a_tile_stride);
                load_global_to_bufferReg(ldg_b_reg, B, BK, BN, K, N, b_tile_row + next_k, b_tile_col, b_tile_stride);
            }

            int load_index_reg = 0;
            for (int bk = 0; bk < BK; bk++) {
                load_index_reg ^= 1;
                if (bk + 1 < BK) {
                    load_shared_to_reg(a_frag[load_index_reg], As[write_index ^ 1], BK, BM, TM, bk + 1, ty);
                    load_shared_to_reg(b_frag[load_index_reg], Bs[write_index ^ 1], BK, BN, TN, bk + 1, tx);
                }
                for (int m = 0; m < TM; m++) {
                    for (int n = 0; n < TN; n++) {
                        accum[m][n] += a_frag[load_index_reg ^ 1][m] * b_frag[load_index_reg ^ 1][n];
                    }
                }
            }
            
            if (next_k < k) {
                /*load_bufferReg_to_shared_a(bufferReg_global_to_shared_a, shared_a[load_index_shared], 
                    block_tile_m, block_tile_k, M, K, global_start_row_a, global_start_col_a, global_stride_a);
                load_bufferReg_to_shared_b(bufferReg_global_to_shared_b, shared_b[load_index_shared], 
                    block_tile_k, block_tile_n, K, N, global_start_row_b, global_start_col_b, global_stride_b);*/

                load_bufferReg_to_shared_a(ldg_a_reg, As[write_index], BM, BK, M, K, a_tile_row, a_tile_col, a_tile_stride);
                load_bufferReg_to_shared_b(ldg_b_reg, Bs[write_index], BK, BN, K, N, b_tile_row, b_tile_col, b_tile_stride);

                __syncthreads();

                load_shared_to_reg(shared_to_reg_a[0], shared_a[load_index_shared], 
                    block_tile_k, block_tile_m, thread_tile_m, 0, thread_y);
                load_shared_to_reg(shared_to_reg_b[0], shared_b[load_index_shared], 
                    block_tile_k, block_tile_n, thread_tile_n, 0, thread_x);
            }
        }

        for (int i = 0; i < thread_tile_m; ++i) {
            for (int j = 0; j < thread_tile_n; ++j) {
                float &accum_ref = accum[i][j];
                float &C_ref = *(C + (thread_y + i) * N + (thread_x + j));
                
                accum_ref = alpha * accum_ref + beta * C_ref;
                C_ref = accum_ref;
            }
        }
    }

    void test_mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
        sgemm<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    int bytes_A = sizeof(float) * M * K;
    int bytes_B = sizeof(float) * K * N;
    int bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);
    float* h_C2 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1;
    }

    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
        h_C1[i] = 0.0f;
    }

    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        gemm_computational_opt::test_mysgemm_v7(M, N, K, 1.0, d_A, d_B, 0.0, d_C);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n", gigaFlops[0], msecPerMatrixMul[0], flopsPerMatrixMul);

    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy(d_C, h_C1, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n", gigaFlops[1], msecPerMatrixMul[1], flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-3;  
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C1[i]) > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], h_C1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}