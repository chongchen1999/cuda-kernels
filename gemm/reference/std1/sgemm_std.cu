// optimize sgemm

#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

__device__ __forceinline__ void load_global_to_register(float * const &buffer, const float * const &A, 
    const int &bm, const int &bn, const int &M, const int &N, 
        const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        *(reinterpret_cast<float4 *>(buffer) + offset) = 
            *(reinterpret_cast<const float4 *>(A + (start_row + i) * N + start_col));
    }
}

__device__ __forceinline__ void load_register_to_shared_a(const float * const &buffer, float * const &shared, 
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

__device__ __forceinline__ void load_register_to_shared_b(const float * const &buffer, float * const &shared, 
    const int &bm, const int &bn, const int &M, const int &N, 
        const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        *(reinterpret_cast<float4 *>(shared + (start_row + i) * bn + start_col)) = 
            *(reinterpret_cast<const float4 *>(buffer) + offset);
    }
}

__device__ __forceinline__ void load_shared_to_register(float * const &reg, const float * const &shared, 
    const int &bm, const int &bn, const int &len, 
        const int &offset_row, const int &offset_col) {
    for (int i = 0; i < len; i += 4) {
        *(reinterpret_cast<float4 *>(reg + i)) = 
            *(reinterpret_cast<const float4 *>(shared + offset_row * bn + (offset_col + i)));
    }
}

namespace gemm_computational_opt {
    #define block_tile_m BM
    #define block_tile_n BN
    #define block_tile_k BK
    #define thread_tile_m TM
    #define thread_tile_n TN
    #define shared_to_reg_a a_frag
    #define shared_to_reg_b b_frag
    #define bufferReg_global_to_shared_a ldg_a_reg
    #define bufferReg_global_to_shared_b ldg_b_reg
    #define loads_per_thread_a ldg_a_num
    #define loads_per_thread_b ldg_b_num
    
    #define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

    template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
    __global__ void mysgemm_v7(int M, int N, int K, float alpha, 
        float *device_A, float *device_B, float beta, float *device_C) {
        const int tid = threadIdx.x;

        /*const int thread_processed_tile_x_dim = BN / TN;
        const int thread_processed_tile_y_dim = BM / TM;
        const int thread_num = thread_processed_tile_x_dim * thread_processed_tile_y_dim; // 一个线程负责计算block中TM*TN个元素
        int tx = (threadIdx.x % thread_processed_tile_x_dim) * TN;
        int ty = (threadIdx.x / thread_processed_tile_x_dim) * TM;
        const int ldg_a_num = BK * BM / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至As需要所有线程搬运ldg_a_num轮
        const int ldg_b_num = BK * BN / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至Bs需要所有线程搬运ldg_b_num轮*/

        const int thread_processed_tile_x_dim = block_tile_n / thread_tile_n;
        const int thread_processed_tile_y_dim = block_tile_m / thread_tile_m;
        const int thread_num = thread_processed_tile_x_dim * thread_processed_tile_y_dim;
        const int tx = threadIdx.x % thread_processed_tile_x_dim * TN;
        const int ty = threadIdx.x / thread_processed_tile_x_dim * TM;
        const int ldg_a_num = block_tile_m * block_tile_k / (thread_num << 2);
        const int ldg_b_num = block_tile_k * block_tile_n / (thread_num << 2);

        /*int a_tile_row = threadIdx.x / (BK / 4); // 每行4个字节作为一个内存块，当前线程负责第a_tile_row行的第a_tile_col个内存块的搬运
        int a_tile_col = threadIdx.x % (BK / 4) * 4;
        int a_tile_stride = BM / ldg_a_num; // 一共BM行，搬运ldg_a_num轮，每论搬运a_tile_stride行

        int b_tile_row = threadIdx.x / (BN / 4); // 每行4个字节作为一个内存块，当前线程负责第b_tile_row行的第b_tile_col个内存块的搬运
        int b_tile_col = threadIdx.x % (BN / 4) * 4;
        int b_tile_stride = BK / ldg_b_num; // 一共BK行，搬运ldg_b_num轮，每论搬运b_tile_stride行*/

        int a_tile_row = tid / (block_tile_k >> 2);
        int a_tile_col = (tid % (block_tile_k >> 2)) << 2;
        int a_tile_stride = block_tile_m / loads_per_thread_a;

        int b_tile_row = tid / (block_tile_n >> 2);
        int b_tile_col = (tid % (block_tile_n >> 2)) << 2;
        int b_tile_stride = block_tile_k / loads_per_thread_b;

        // __shared__ float As[2][BK * BM]; // 增加一倍共享内存大小用于缓存
        // __shared__ float Bs[2][BK * BN]; // 增加一倍共享内存大小用于缓存
        // float accum[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；
        __shared__ float As[2][block_tile_m * block_tile_k];
        __shared__ float Bs[2][block_tile_k * block_tile_n];
        float accum[thread_tile_m][thread_tile_n] = {0.0};

        // 计算ldg_a_num的所有参数必须全部是const，否则不能用来申明数组大小
        // float ldg_a_reg[4 * ldg_a_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵
        // float ldg_b_reg[4 * ldg_b_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵
        float bufferReg_global_to_shared_a[loads_per_thread_a << 2] = {0.0};
        float bufferReg_global_to_shared_b[loads_per_thread_b << 2] = {0.0};

        // float a_frag[2][TM];  // 缓存As共享内存,增加一倍寄存器大小用于缓存
        // float b_frag[2][TN];  // 缓存Bs共享内存,增加一倍寄存器大小用于缓存
        float shared_to_reg_a[2][thread_tile_m];
        float shared_to_reg_b[2][thread_tile_n];

        // 移动到当前block
        const float *A = device_A + blockIdx.y * block_tile_m * K;
        const float *B = device_B + blockIdx.x * block_tile_n;
        float *C = device_C + blockIdx.y * block_tile_m * N + blockIdx.x * block_tile_n;

        // first global to shared
        /*
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
            As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 不需要转置
        }*/


        // #pragma unroll
        /*for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        }

        // #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            int ldg_index = i / b_tile_stride * 4;  // 第ldg_index轮
            FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }*/

        load_global_to_register(ldg_a_reg, A, BM, BK, M, K, a_tile_row, a_tile_col, a_tile_stride);
        load_global_to_register(ldg_b_reg, B, BK, BN, K, N, b_tile_row, b_tile_col, b_tile_stride);

        /*for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;
            As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
        //#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            int ldg_index = i / b_tile_stride * 4;
            FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(ldg_b_reg[ldg_index]);
        }*/

        load_register_to_shared_a(ldg_a_reg, As[0], BM, BK, M, K, a_tile_row, a_tile_col, a_tile_stride);
        load_register_to_shared_b(ldg_b_reg, Bs[0], BK, BN, K, N, b_tile_row, b_tile_col, b_tile_stride);

        __syncthreads();

        // first shared to frag
        //#pragma unroll
        /*for (int m = 0; m < TM; m += 4) {
            FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]); // 偏移到当前thread tile
        }
        //#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]); // 偏移到当前thread tile
        }*/

        load_shared_to_register(a_frag[0], As[0], BK, BM, TM, 0, ty);
        load_shared_to_register(b_frag[0], Bs[0], BK, BN, TN, 0, tx);

        int write_index = 0;
        for (int k = 0; k < K; k += BK) {
            write_index ^= 1;
            int next_k = k + BK;
            // load global to reg
            if (next_k < K) {
                // #pragma unroll
                /*for (int i = 0; i < BM; i += a_tile_stride) {
                    int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
                    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, next_k + a_tile_col, K)]);
                }
                // #pragma unroll
                for (int i = 0; i < BK; i += b_tile_stride) {
                    int ldg_index = i / b_tile_stride * 4;  // 第ldg_index轮
                    FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                            FETCH_FLOAT4(B[OFFSET(next_k + b_tile_row + i, b_tile_col, N)]);
                }*/
                load_global_to_register(ldg_a_reg, A, BM, BK, M, K, a_tile_row, a_tile_col + next_k, a_tile_stride);
                load_global_to_register(ldg_b_reg, B, BK, BN, K, N, b_tile_row + next_k, b_tile_col, b_tile_stride);
            }

            int load_index_reg = 0;
            // #pragma unroll
            for (int bk = 0; bk < BK; bk++) {
                load_index_reg ^= 1;
                if (bk + 1 < BK) {
                    /*for (int m = 0; m < TM; m += 4) {
                        FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                                As[load_index][OFFSET(bk + 1, ty + m, BM)]); // 偏移到当前thread tile
                    }
                    // #pragma unroll
                    for (int n = 0; n < TN; n += 4) {
                        FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                                Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); // 偏移到当前thread tile
                    }*/
                    load_shared_to_register(a_frag[load_index_reg], As[write_index ^ 1], BK, BM, TM, bk + 1, ty);
                    load_shared_to_register(b_frag[load_index_reg], Bs[write_index ^ 1], BK, BN, TN, bk + 1, tx);
                }
                // #pragma unroll
                for (int m = 0; m < TM; m++) {
                    for (int n = 0; n < TN; n++) {
                        accum[m][n] += a_frag[load_index_reg ^ 1][m] * b_frag[load_index_reg ^ 1][n];
                    }
                }
            }

            if (next_k < K) {
                //#pragma unroll
                /*for (int i = 0; i < BM; i += a_tile_stride) {
                    int ldg_index = i / a_tile_stride * 4;
                    As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                    As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                    As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                    As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
                }
                //#pragma unroll
                for (int i = 0; i < BK; i += b_tile_stride) {
                    int ldg_index = i / b_tile_stride * 4;
                    FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                            FETCH_FLOAT4(ldg_b_reg[ldg_index]);
                }*/

                load_register_to_shared_a(ldg_a_reg, As[write_index], BM, BK, M, K, a_tile_row, a_tile_col, a_tile_stride);
                load_register_to_shared_b(ldg_b_reg, Bs[write_index], BK, BN, K, N, b_tile_row, b_tile_col, b_tile_stride);

                __syncthreads();
                //#pragma unroll
                /*for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(
                            As[write_index][OFFSET(0, ty + m, BM)]); // 偏移到当前thread tile
                }
                //#pragma unroll
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(
                            Bs[write_index][OFFSET(0, tx + n, BN)]); // 偏移到当前thread tile
                }*/
                load_shared_to_register(a_frag[0], As[write_index], BK, BM, TM, 0, ty);
                load_shared_to_register(b_frag[0], Bs[write_index], BK, BN, TN, 0, tx);
            }
        }
        
        // C = alpha*AB+C
        //#pragma unroll
        for (int i = 0; i < TM; i++) {
            //#pragma unroll
            for (int j = 0; j < TN; j++) {
                /*float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
                ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
                ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
                ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
                ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
                FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;*/
                float &accum_ref = accum[i][j];
                float &C_ref = *(C + (ty + i) * N + (tx + j));
            
                accum_ref = alpha * accum_ref + beta * C_ref;
                C_ref = accum_ref;
            }
        }
    }

    void test_mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
        mysgemm_v7<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    }
}
// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void Sgemm( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    // registers load global memory
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_NUM_PER_BLOCK * 4);
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; 
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    //transfer first tile from global mem to shared mem
    // load A from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }
    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL, // col
                N )]);
    }
    __syncthreads();
    // load A from shared memory to register
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile from global mem
        if(tile_idx< K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL, // col
                    N )]);
            }
        }

        int load_stage_idx = write_stage_idx ^ 1;

        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            // load next tile from shared mem to register 
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }

        // load first tile from shared mem to register of next iter
        // load A from shared memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //compute last tile mma THREAD_SIZE_X x THREAD_SIZE_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx< K);

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
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

    // assert( M%8 == 0); 
    // assert( N%8 == 0); 
    // assert( K%8 == 0); 

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

    /*const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;*/

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i % 3;
        h_A[i] = 1;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = ((long long)i * i) % 3;
        h_B[i] = 1;
    }

    for( int i = 0; i < M * N; i++ ) {
        h_C[i] = 0.0f;
        h_C1[i] = 0.0f;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        /*Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);*/

        // gemm::OptsGemm(M, N, K, d_A, d_B, d_C, 1.0, 0);
        // int M, int N, int K, float alpha, float *A, float *B, float beta, float *C
        gemm_computational_opt::test_mysgemm_v7(M, N, K, 1.0, d_A, d_B, 0.0, d_C);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    /*for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0;
            for (int k = 0; k < K; k++) {
                temp += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C2[i * N + j] = temp;
        }
    }*/

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // return 0;

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C1, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        /*cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );*/
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B,
                        N, d_A, K, &beta, d_C, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 
    
    double eps = 1.e-3;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C1[i]) > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}