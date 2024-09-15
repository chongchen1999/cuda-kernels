// A100 PCIE 80GB
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 4.46636ms
// TFLOPS: 26.5048

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include "includes/commons.cuh"

const int bm = 128;
const int bn = 128;
const int bk = 32;

const int wm = 64;
const int wn = 64;
const int wk = 16;

const int wmma_m = 16;
const int wmma_n = 16;
const int wmma_k = 16;

__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko) {
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * warpSize * 2 + ty * warpSize + tx;

    const int load_iters = M / (ty * tz);
    for (int i = 0; i < load_iters; ++i) {
        const int row = i * 4 + tid / warpSize;
        const int col = tid % warpSize;

        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = 
            A[(by * block_tile_m + row) * K + ko * bk + col];
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko) {
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * warpSize * 2 + ty * warpSize + tx;

    const int load_iters = N / (ty * tz);
    for (int i = 0; i < load_iters; ++i) {
        const int row = i * 4 + tid / warpSize;
        const int col = tid % warpSize;

        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = 
            B[(bx * block_tile_n + row) * K + ko * bk + col];
    }
}

__device__ void loadSmemC(float *smem, half *C, int M, int N) {
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i) {
        const int row = i;
        const int col = tid;

        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = 
            (float)(C[(by * 128 + row) * N + bx * 128 + col]);
    }
}

__device__ void storeSmemC(half *C, float *smem, int M, int N) {
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i) {
        const int row = i;
        const int col = tid;

        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        C[(by * 128 + row) * N + bx * 128 + col] = 
            *reinterpret_cast<half *>(smem + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
    }
}

__device__ void loadFragA(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, nvcuda::wmma::row_major> *frag, 
    half *smem, 
    int ki
) {
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
        int row = tz * 64 + i * 16;
        int col = ki * wk;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, nvcuda::wmma::col_major> *frag, 
    half *smem, 
    int ki
) {
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i) {
        int row = ty * 64 + i * 16;
        int col = ki * wk;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void storeAccum(
    float *ptr, 
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmma_m, wmma_n, wmma_k, float> *frag
) {
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(
                ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), 
                frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major
            );
        }
    }
}

__global__ void matmul(
    half *A, half *B, half *C, 
    int M, int N, int K, 
    float alpha, float beta
) {
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ char shared_storage[];
    half *SA = reinterpret_cast<half *>(shared_storage);
    half *SB = reinterpret_cast<half *>(shared_storage + bm * bk * sizeof(half));
    float *SC = reinterpret_cast<float *>(shared_storage);

    const int frag_num_m = wm / wmma_m;
    const int frag_num_n = wn / wmma_n;
    const int frag_num_accum = frag_num_m * frag_num_n;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, nvcuda::wmma::row_major> frag_a[frag_num_m];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, nvcuda::wmma::col_major> frag_b[frag_num_n];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmma_m, wmma_n, wmma_k, float> accum[frag_num_accum];

    for (int i = 0; i < frag_num_m; ++i) {
        for (int j = 0; j < frag_num_n; ++j) {
            nvcuda::wmma::fill_fragment(accum[i * frag_num_n + j], 0.0);
        }
    }

    for (int ko = 0; ko < K / bk; ++ko) {
        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, N, K, ko);
        __syncthreads();

        for (int ki = 0; ki < bk / wk; ++ki) {
            // 64x64x16 mma for each warp
            loadFragA(frag_a, SA, ki);
            loadFragB(frag_b, SB, ki);

            for (int i = 0; i < frag_num_m; ++i) {
                for (int j = 0; j < frag_num_n; ++j) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(
                        accum[i * frag_num_n + j], 
                        frag_a[i], frag_b[j], 
                        accum[i * frag_num_n + j]
                    );
                }
            }
        }
    }

    storeAccum(SC, accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}