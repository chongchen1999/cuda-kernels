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

namespace gemm {

    #define BLOCK_X 16
    #define BLOCK_Y 16

    #define TILE_X 128
    #define TILE_X_4 32
    #define TILE_Y 128
    #define TILE_Y_4 32

    #define TILE_K 16

    #define WPTN 8
    #define WPTM 8
    #define WPTN_4 2

    __global__ void gemm_kernel_NN(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float4* __restrict__ C,
        float alpha, float beta,
        int M, int N, int K)
    {
        __shared__ float4 smem_a[2][TILE_K * TILE_Y_4];
        __shared__ float4 smem_b[2][TILE_K * TILE_X_4];

        int tx = threadIdx.x % 16;
        int ty = threadIdx.x / 16;
        int tx4 = threadIdx.x % 4;
        int ty4 = threadIdx.x / 4;
        int tx32 = threadIdx.x % 32;
        int ty32 = threadIdx.x / 32;

        const float* pA = A + K * TILE_Y * blockIdx.y + ty4 * K + tx4 * 4;
        const float* pB = B + TILE_X * blockIdx.x + ty32 * N + tx32 * 4;
        float4* pC = C + TILE_Y * blockIdx.y * N / 4 + TILE_X_4 * blockIdx.x;

        int sts_a_offset = tx4 * 4 * TILE_Y + ty4;
        int sts_b_offset = ty32 * TILE_X_4 + tx32;

        float4 f4_zero = make_float4(0.f, 0.f, 0.f, 0.f);
        bool valid_ld_a_0 = ((blockIdx.y * TILE_Y + ty4) < M) && ((tx4 * 4) < K);
        bool valid_ld_a_1 = ((blockIdx.y * TILE_Y + ty4 + 64) < M) && ((tx4 * 4) < K); 
        bool valid_ld_b_0 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && (ty32 < K);
        bool valid_ld_b_1 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && ((ty32 + 8) < K);

        float4 ldg_a_reg[2];
        float4 ldg_b_reg[2];

        ldg_a_reg[0] = valid_ld_a_0 ? *(const float4*)pA : f4_zero;
        ldg_a_reg[1] = valid_ld_a_1 ? *(const float4*)(pA + 64 * K) : f4_zero;
        ldg_b_reg[0] = valid_ld_b_0 ? *(const float4*)(pB + 0 * N) : f4_zero;
        ldg_b_reg[1] = valid_ld_b_1 ? *(const float4*)(pB + 8 * N) : f4_zero;

        float4 c[WPTM][WPTN_4] = { { f4_zero } };

        *((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
        *((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
        *((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
        *((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
        *((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
        *((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
        *((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
        *((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

        smem_b[0][sts_b_offset + 0] = ldg_b_reg[0];
        smem_b[0][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];

        __syncthreads();

        int i = 0;
        int write_stage_idx = 1;

        float4 reg_a[2][2];
        float4 reg_b[2][2];

        reg_a[0][0] = smem_a[0][0 + ty];
        reg_a[0][1] = smem_a[0][16 + ty];
        reg_b[0][0] = smem_b[0][0 + tx];
        reg_b[0][1] = smem_b[0][16 + tx];

        do {
            i += 16;
            valid_ld_a_0 = (valid_ld_a_0 && ((tx4 * 4 + i) < K));
            valid_ld_a_1 = (valid_ld_a_1 && ((tx4 * 4 + i) < K));
            valid_ld_b_0 = (valid_ld_b_0 && ((ty32 + i) < K));
            valid_ld_b_1 = (valid_ld_b_1 && ((ty32 + 8 + i) < K));

            ldg_a_reg[0] = (valid_ld_a_0) ? *(const float4*)(pA + i + 0) : f4_zero;
            ldg_a_reg[1] = (valid_ld_a_1) ? *(const float4*)(pA + i + 64 * K) : f4_zero;
            ldg_b_reg[0] = (valid_ld_b_0) ? *(const float4*)(pB + (i + 0) * N) : f4_zero;
            ldg_b_reg[1] = (valid_ld_b_1) ? *(const float4*)(pB + (i + 8) * N) : f4_zero;

            int load_stage_idx = write_stage_idx ^ 1;

    #pragma unroll
            for (int j = 0; j < TILE_K - 1; j++) {
                reg_a[(j + 1) % 2][0] = smem_a[load_stage_idx][(j + 1) * TILE_Y_4 + 0 + ty];
                reg_a[(j + 1) % 2][1] = smem_a[load_stage_idx][(j + 1) * TILE_Y_4 + 16 + ty];
                reg_b[(j + 1) % 2][0] = smem_b[load_stage_idx][(j + 1) * TILE_X_4 + 0 + tx];
                reg_b[(j + 1) % 2][1] = smem_b[load_stage_idx][(j + 1) * TILE_X_4 + 16 + tx];

                c[0][0].x += reg_a[j & 1][0].x * reg_b[j & 1][0].x;
                c[0][0].y += reg_a[j & 1][0].x * reg_b[j & 1][0].y;
                c[0][0].z += reg_a[j & 1][0].x * reg_b[j & 1][0].z;
                c[0][0].w += reg_a[j & 1][0].x * reg_b[j & 1][0].w;
                c[0][1].x += reg_a[j & 1][0].x * reg_b[j & 1][1].x;
                c[0][1].y += reg_a[j & 1][0].x * reg_b[j & 1][1].y;
                c[0][1].z += reg_a[j & 1][0].x * reg_b[j & 1][1].z;
                c[0][1].w += reg_a[j & 1][0].x * reg_b[j & 1][1].w;

                c[1][0].x += reg_a[j & 1][0].y * reg_b[j & 1][0].x;
                c[1][0].y += reg_a[j & 1][0].y * reg_b[j & 1][0].y;
                c[1][0].z += reg_a[j & 1][0].y * reg_b[j & 1][0].z;
                c[1][0].w += reg_a[j & 1][0].y * reg_b[j & 1][0].w;
                c[1][1].x += reg_a[j & 1][0].y * reg_b[j & 1][1].x;
                c[1][1].y += reg_a[j & 1][0].y * reg_b[j & 1][1].y;
                c[1][1].z += reg_a[j & 1][0].y * reg_b[j & 1][1].z;
                c[1][1].w += reg_a[j & 1][0].y * reg_b[j & 1][1].w;

                c[2][0].x += reg_a[j & 1][0].z * reg_b[j & 1][0].x;
                c[2][0].y += reg_a[j & 1][0].z * reg_b[j & 1][0].y;
                c[2][0].z += reg_a[j & 1][0].z * reg_b[j & 1][0].z;
                c[2][0].w += reg_a[j & 1][0].z * reg_b[j & 1][0].w;
                c[2][1].x += reg_a[j & 1][0].z * reg_b[j & 1][1].x;
                c[2][1].y += reg_a[j & 1][0].z * reg_b[j & 1][1].y;
                c[2][1].z += reg_a[j & 1][0].z * reg_b[j & 1][1].z;
                c[2][1].w += reg_a[j & 1][0].z * reg_b[j & 1][1].w;

                c[3][0].x += reg_a[j & 1][0].w * reg_b[j & 1][0].x;
                c[3][0].y += reg_a[j & 1][0].w * reg_b[j & 1][0].y;
                c[3][0].z += reg_a[j & 1][0].w * reg_b[j & 1][0].z;
                c[3][0].w += reg_a[j & 1][0].w * reg_b[j & 1][0].w;
                c[3][1].x += reg_a[j & 1][0].w * reg_b[j & 1][1].x;
                c[3][1].y += reg_a[j & 1][0].w * reg_b[j & 1][1].y;
                c[3][1].z += reg_a[j & 1][0].w * reg_b[j & 1][1].z;
                c[3][1].w += reg_a[j & 1][0].w * reg_b[j & 1][1].w;

                c[4][0].x += reg_a[j & 1][1].x * reg_b[j & 1][0].x;
                c[4][0].y += reg_a[j & 1][1].x * reg_b[j & 1][0].y;
                c[4][0].z += reg_a[j & 1][1].x * reg_b[j & 1][0].z;
                c[4][0].w += reg_a[j & 1][1].x * reg_b[j & 1][0].w;
                c[4][1].x += reg_a[j & 1][1].x * reg_b[j & 1][1].x;
                c[4][1].y += reg_a[j & 1][1].x * reg_b[j & 1][1].y;
                c[4][1].z += reg_a[j & 1][1].x * reg_b[j & 1][1].z;
                c[4][1].w += reg_a[j & 1][1].x * reg_b[j & 1][1].w;

                c[5][0].x += reg_a[j & 1][1].y * reg_b[j & 1][0].x;
                c[5][0].y += reg_a[j & 1][1].y * reg_b[j & 1][0].y;
                c[5][0].z += reg_a[j & 1][1].y * reg_b[j & 1][0].z;
                c[5][0].w += reg_a[j & 1][1].y * reg_b[j & 1][0].w;
                c[5][1].x += reg_a[j & 1][1].y * reg_b[j & 1][1].x;
                c[5][1].y += reg_a[j & 1][1].y * reg_b[j & 1][1].y;
                c[5][1].z += reg_a[j & 1][1].y * reg_b[j & 1][1].z;
                c[5][1].w += reg_a[j & 1][1].y * reg_b[j & 1][1].w;

                c[6][0].x += reg_a[j & 1][1].z * reg_b[j & 1][0].x;
                c[6][0].y += reg_a[j & 1][1].z * reg_b[j & 1][0].y;
                c[6][0].z += reg_a[j & 1][1].z * reg_b[j & 1][0].z;
                c[6][0].w += reg_a[j & 1][1].z * reg_b[j & 1][0].w;
                c[6][1].x += reg_a[j & 1][1].z * reg_b[j & 1][1].x;
                c[6][1].y += reg_a[j & 1][1].z * reg_b[j & 1][1].y;
                c[6][1].z += reg_a[j & 1][1].z * reg_b[j & 1][1].z;
                c[6][1].w += reg_a[j & 1][1].z * reg_b[j & 1][1].w;

                c[7][0].x += reg_a[j & 1][1].w * reg_b[j & 1][0].x;
                c[7][0].y += reg_a[j & 1][1].w * reg_b[j & 1][0].y;
                c[7][0].z += reg_a[j & 1][1].w * reg_b[j & 1][0].z;
                c[7][0].w += reg_a[j & 1][1].w * reg_b[j & 1][0].w;
                c[7][1].x += reg_a[j & 1][1].w * reg_b[j & 1][1].x;
                c[7][1].y += reg_a[j & 1][1].w * reg_b[j & 1][1].y;
                c[7][1].z += reg_a[j & 1][1].w * reg_b[j & 1][1].z;
                c[7][1].w += reg_a[j & 1][1].w * reg_b[j & 1][1].w;
            }

            if (i < K) {
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
                *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

                smem_b[write_stage_idx][sts_b_offset + 0] = ldg_b_reg[0];
                smem_b[write_stage_idx][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];
                __syncthreads();
                write_stage_idx ^= 1;
            }

            reg_a[0][0] = smem_a[load_stage_idx ^ 1][0 + ty];
            reg_a[0][1] = smem_a[load_stage_idx ^ 1][16 + ty];
            reg_b[0][0] = smem_b[load_stage_idx ^ 1][0 + tx];
            reg_b[0][1] = smem_b[load_stage_idx ^ 1][16 + tx];

            c[0][0].x += reg_a[1][0].x * reg_b[1][0].x;
            c[0][0].y += reg_a[1][0].x * reg_b[1][0].y;
            c[0][0].z += reg_a[1][0].x * reg_b[1][0].z;
            c[0][0].w += reg_a[1][0].x * reg_b[1][0].w;
            c[0][1].x += reg_a[1][0].x * reg_b[1][1].x;
            c[0][1].y += reg_a[1][0].x * reg_b[1][1].y;
            c[0][1].z += reg_a[1][0].x * reg_b[1][1].z;
            c[0][1].w += reg_a[1][0].x * reg_b[1][1].w;

            c[1][0].x += reg_a[1][0].y * reg_b[1][0].x;
            c[1][0].y += reg_a[1][0].y * reg_b[1][0].y;
            c[1][0].z += reg_a[1][0].y * reg_b[1][0].z;
            c[1][0].w += reg_a[1][0].y * reg_b[1][0].w;
            c[1][1].x += reg_a[1][0].y * reg_b[1][1].x;
            c[1][1].y += reg_a[1][0].y * reg_b[1][1].y;
            c[1][1].z += reg_a[1][0].y * reg_b[1][1].z;
            c[1][1].w += reg_a[1][0].y * reg_b[1][1].w;

            c[2][0].x += reg_a[1][0].z * reg_b[1][0].x;
            c[2][0].y += reg_a[1][0].z * reg_b[1][0].y;
            c[2][0].z += reg_a[1][0].z * reg_b[1][0].z;
            c[2][0].w += reg_a[1][0].z * reg_b[1][0].w;
            c[2][1].x += reg_a[1][0].z * reg_b[1][1].x;
            c[2][1].y += reg_a[1][0].z * reg_b[1][1].y;
            c[2][1].z += reg_a[1][0].z * reg_b[1][1].z;
            c[2][1].w += reg_a[1][0].z * reg_b[1][1].w;

            c[3][0].x += reg_a[1][0].w * reg_b[1][0].x;
            c[3][0].y += reg_a[1][0].w * reg_b[1][0].y;
            c[3][0].z += reg_a[1][0].w * reg_b[1][0].z;
            c[3][0].w += reg_a[1][0].w * reg_b[1][0].w;
            c[3][1].x += reg_a[1][0].w * reg_b[1][1].x;
            c[3][1].y += reg_a[1][0].w * reg_b[1][1].y;
            c[3][1].z += reg_a[1][0].w * reg_b[1][1].z;
            c[3][1].w += reg_a[1][0].w * reg_b[1][1].w;

            c[4][0].x += reg_a[1][1].x * reg_b[1][0].x;
            c[4][0].y += reg_a[1][1].x * reg_b[1][0].y;
            c[4][0].z += reg_a[1][1].x * reg_b[1][0].z;
            c[4][0].w += reg_a[1][1].x * reg_b[1][0].w;
            c[4][1].x += reg_a[1][1].x * reg_b[1][1].x;
            c[4][1].y += reg_a[1][1].x * reg_b[1][1].y;
            c[4][1].z += reg_a[1][1].x * reg_b[1][1].z;
            c[4][1].w += reg_a[1][1].x * reg_b[1][1].w;

            c[5][0].x += reg_a[1][1].y * reg_b[1][0].x;
            c[5][0].y += reg_a[1][1].y * reg_b[1][0].y;
            c[5][0].z += reg_a[1][1].y * reg_b[1][0].z;
            c[5][0].w += reg_a[1][1].y * reg_b[1][0].w;
            c[5][1].x += reg_a[1][1].y * reg_b[1][1].x;
            c[5][1].y += reg_a[1][1].y * reg_b[1][1].y;
            c[5][1].z += reg_a[1][1].y * reg_b[1][1].z;
            c[5][1].w += reg_a[1][1].y * reg_b[1][1].w;

            c[6][0].x += reg_a[1][1].z * reg_b[1][0].x;
            c[6][0].y += reg_a[1][1].z * reg_b[1][0].y;
            c[6][0].z += reg_a[1][1].z * reg_b[1][0].z;
            c[6][0].w += reg_a[1][1].z * reg_b[1][0].w;
            c[6][1].x += reg_a[1][1].z * reg_b[1][1].x;
            c[6][1].y += reg_a[1][1].z * reg_b[1][1].y;
            c[6][1].z += reg_a[1][1].z * reg_b[1][1].z;
            c[6][1].w += reg_a[1][1].z * reg_b[1][1].w;

            c[7][0].x += reg_a[1][1].w * reg_b[1][0].x;
            c[7][0].y += reg_a[1][1].w * reg_b[1][0].y;
            c[7][0].z += reg_a[1][1].w * reg_b[1][0].z;
            c[7][0].w += reg_a[1][1].w * reg_b[1][0].w;
            c[7][1].x += reg_a[1][1].w * reg_b[1][1].x;
            c[7][1].y += reg_a[1][1].w * reg_b[1][1].y;
            c[7][1].z += reg_a[1][1].w * reg_b[1][1].z;
            c[7][1].w += reg_a[1][1].w * reg_b[1][1].w;

        } while (i < K);

    #pragma unroll
        for (int wm = 0; wm < WPTM; wm++) {
    #pragma unroll
            for (int wn = 0; wn < WPTN_4; wn++) {
                c[wm][wn].x *= alpha;
                c[wm][wn].y *= alpha;
                c[wm][wn].z *= alpha;
                c[wm][wn].w *= alpha;
            }
        }

    #pragma unroll
        for (int wm = 0; wm < 4; wm++) {
    #pragma unroll
            for (int wn = 0; wn < WPTN_4; wn++) {
                if (((blockIdx.y * TILE_Y + ty * 4 + wm) < M) 
                    && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
                    if (beta != 0) {
                        float4 vec4c = *(pC + ((ty * 4 + wm) * N / 4 + wn * 16 + tx));
                        vec4c.x = vec4c.x * beta + c[wm][wn].x;
                        vec4c.y = vec4c.y * beta + c[wm][wn].y;
                        vec4c.z = vec4c.z * beta + c[wm][wn].z;
                        vec4c.w = vec4c.w * beta + c[wm][wn].w;
                        *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                    } else {
                        *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm][wn];
                    }
                }
            }
        }

    #pragma unroll
        for (int wm = 0; wm < 4; wm++) {
    #pragma unroll
            for (int wn = 0; wn < WPTN_4; wn++) {
                if (((blockIdx.y * TILE_Y + 64 + ty * 4 + wm) < M) 
                    && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
                    if (beta != 0) {
                        float4 vec4c = *(pC + ((64 + ty * 4 + wm) * N / 4 + wn * 16 + tx));
                        vec4c.x = vec4c.x * beta + c[wm + 4][wn].x;
                        vec4c.y = vec4c.y * beta + c[wm + 4][wn].y;
                        vec4c.z = vec4c.z * beta + c[wm + 4][wn].z;
                        vec4c.w = vec4c.w * beta + c[wm + 4][wn].w;
                        *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                    } else {
                        *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm + 4][wn];
                    }
                }
            }
        }
    }

    void OptsGemm(int m, int n, int k, float* d_A, float* d_B, float* d_C, float alpha, float beta) {
        constexpr int BLOCK_DIM = 128;
        dim3 block(256);
        dim3 grid((m + BLOCK_DIM - 1) / BLOCK_DIM, (n + BLOCK_DIM - 1) / BLOCK_DIM);

        gemm_kernel_NN<<<grid, block>>>(d_A, d_B, (float4*)d_C, alpha, beta, m, n, k);
    }
}

namespace gemm_computational_opt {
    #define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

    template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
    __global__ void mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
        int bx = blockIdx.x;
        int by = blockIdx.y;

        const int thread_processed_tile_x_dim = BN / TN;
        const int thread_processed_tile_y_dim = BM / TM;
        const int thread_num = thread_processed_tile_x_dim * thread_processed_tile_y_dim; // 一个线程负责计算block中TM*TN个元素

        // 当前线程对应thread tile的左上角元素在block中的位置
        int tx = (threadIdx.x % thread_processed_tile_x_dim) * TN;
        int ty = (threadIdx.x / thread_processed_tile_x_dim) * TM;

        __shared__ float As[2][BK * BM]; // 增加一倍共享内存大小用于缓存
        __shared__ float Bs[2][BK * BN];


        const int ldg_a_num = BK * BM / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至As需要所有线程搬运ldg_a_num轮
        const int ldg_b_num = BK * BN / thread_num / 4; // 每个线程搬运4个浮点数，完成搬运至Bs需要所有线程搬运ldg_b_num轮

        int a_tile_row = threadIdx.x / (BK / 4); // 每行4个字节作为一个内存块，当前线程负责第a_tile_row行的第a_tile_col个内存块的搬运
        int a_tile_col = threadIdx.x % (BK / 4) * 4;
        int a_tile_stride = BM / ldg_a_num; // 一共BM行，搬运ldg_a_num轮，每论搬运a_tile_stride行

        int b_tile_row = threadIdx.x / (BN / 4); // 每行4个字节作为一个内存块，当前线程负责第b_tile_row行的第b_tile_col个内存块的搬运
        int b_tile_col = threadIdx.x % (BN / 4) * 4;
        int b_tile_stride = BK / ldg_b_num; // 一共BK行，搬运ldg_b_num轮，每论搬运b_tile_stride行

        float accum[TM][TN] = {0.}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值，额外的一个寄存器用于缓存；

        // 计算ldg_a_num的所有参数必须全部是const，否则不能用来申明数组大小
        float ldg_a_reg[4 * ldg_a_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵
        float ldg_b_reg[4 * ldg_b_num] = {0.}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵

        float a_frag[2][TM];  // 缓存As共享内存,增加一倍寄存器大小用于缓存
        float b_frag[2][TN];  // 缓存Bs共享内存,增加一倍寄存器大小用于缓存

        // 移动到当前block
        A = &A[by * BM * K];
        B = &B[bx * BN];
        C = &C[by * BM * N + bx * BN];

        // first global to shared
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
        }
        __syncthreads();

        // first shared to frag
    #pragma unroll
        for (int m = 0; m < TM; m += 4) {
            FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]); // 偏移到当前thread tile
        }
    #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]); // 偏移到当前thread tile
        }


        int write_index = 1;
        int load_index;
        int k = 0;
        do {
            k += BK;
            // load global to reg
            if (k < K) {
    #pragma unroll
                for (int i = 0; i < BM; i += a_tile_stride) {
                    int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
                    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
                }
    #pragma unroll
                for (int i = 0; i < BK; i += b_tile_stride) {
                    int ldg_index = i / b_tile_stride * 4;  // 第ldg_index轮
                    FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                            FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
                }
            }

            load_index = write_index ^ 1;
    #pragma unroll
            for (int bk = 0; bk < BK - 1; bk++) {
                for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                            As[load_index][OFFSET(bk + 1, ty + m, BM)]); // 偏移到当前thread tile
                }
    #pragma unroll
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                            Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); // 偏移到当前thread tile
                }
    #pragma unroll
                for (int m = 0; m < TM; m++) {
                    for (int n = 0; n < TN; n++) {
                        accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                    }
                }
            }
            if (k < K) {
    #pragma unroll
                for (int i = 0; i < BM; i += a_tile_stride) {
                    int ldg_index = i / a_tile_stride * 4;
                    As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                    As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                    As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                    As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
                }
    #pragma unroll
                for (int i = 0; i < BK; i += b_tile_stride) {
                    int ldg_index = i / b_tile_stride * 4;
                    FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                            FETCH_FLOAT4(ldg_b_reg[ldg_index]);
                }
                __syncthreads();
    #pragma unroll
                for (int m = 0; m < TM; m += 4) {
                    FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(
                            As[write_index][OFFSET(0, ty + m, BM)]); // 偏移到当前thread tile
                }
    #pragma unroll
                for (int n = 0; n < TN; n += 4) {
                    FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(
                            Bs[write_index][OFFSET(0, tx + n, BN)]); // 偏移到当前thread tile
                }

                write_index ^= 1;
            }
    #pragma unroll
            for (int m = 0; m < TM; m++) {
    #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
                }
            }


        } while (k < K);
        
        // C = alpha*AB+C
    #pragma unroll
        for (int m = 0; m < TM; m++) {
    #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
                ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
                ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
                ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
                ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
                FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
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
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    // assert( M%8 == 0); 
    // assert( N%8 == 0); 
    // assert( K%8 == 0); 

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
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

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i % 3;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = ((long long)i * i) % 3;
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
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
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