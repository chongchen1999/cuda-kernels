// wmma + fake pipeline

// A100 PCIE 80GB
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 3.58903ms
// TFLOPS: 32.9838

// 3090
// Setting to 4 stages.
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 5.69767ms
// TFLOPS: 20.7769

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

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
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i) {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = 
            A[(by * 128 + row) * K + ko * bk + col];
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko) {
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i) {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = 
            B[(bx * 128 + row) * K + ko * bk + col];
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
        int row = i;
        int col = tid;
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
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx * 128 + col]) = 
            (half)smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
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
                frag[i * 4 + j], 16, 
                nvcuda::wmma::mem_row_major
            );
        }
    }
}

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta) {
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    half *SA1 = reinterpret_cast<half *>(shared_storage);
    half *SA2 = SA1 + bm * bk;
    half *SA3 = SA2 + bm * bk;
    half *SA4 = SA3 + bm * bk;
    half *SB1 = SA4 + bm * bk;
    half *SB2 = SB1 + bn * bk;
    half *SB3 = SB2 + bn * bk;
    half *SB4 = SB3 + bn * bk;
    float *SC = reinterpret_cast<float *>(shared_storage);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, nvcuda::wmma::row_major> FragA[wm / wmma_m];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, nvcuda::wmma::col_major> FragB[wn / wmma_n];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmma_m, wmma_n, wmma_k, float> Accum[wm / wmma_m * wn / wmma_n];

    for (int mii = 0; mii < wm / wmma_m; mii += 1) {
        for (int nii = 0; nii < wn / wmma_n; nii += 1) {
            nvcuda::wmma::fill_fragment(Accum[mii * (wn / wmma_n) + nii], 0.0);
        }
    }

    // prologue
    loadSmemA(SA1, A, M, K, 0);
    loadSmemB(SB1, B, N, K, 0);

    loadSmemA(SA2, A, M, K, 1);
    loadSmemB(SB2, B, N, K, 1);

    loadSmemA(SA3, A, M, K, 2);
    loadSmemB(SB3, B, N, K, 2);

    for (int ko = 0; ko < K / bk; ko += 4) {
        __syncthreads();
        if (ko + 3 < K / bk) {
            loadSmemA(SA4, A, M, K, ko + 3);
            loadSmemB(SB4, B, N, K, ko + 3);
        }

        for (int ki = 0; ki < bk / wk; ki += 1) {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA1, ki);
            loadFragB(FragB, SB1, ki);
            for (int mii = 0; mii < wm / wmma_m; mii += 1) {
                for (int nii = 0; nii < wn / wmma_n; nii += 1) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(
                        Accum[mii * (wn / wmma_n) + nii], 
                        FragA[mii], FragB[nii], 
                        Accum[mii * (wn / wmma_n) + nii]
                    );
                }
            }
        }

        __syncthreads();
        if (ko + 4 < K / bk) {
            loadSmemA(SA1, A, M, K, ko + 4);
            loadSmemB(SB1, B, N, K, ko + 4);
        }
        for (int ki = 0; ki < bk / wk; ki += 1) {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA2, ki);
            loadFragB(FragB, SB2, ki);
            for (int mii = 0; mii < wm / wmma_m; mii += 1) {
                for (int nii = 0; nii < wn / wmma_n; nii += 1) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(
                        Accum[mii * (wn / wmma_n) + nii], 
                        FragA[mii], FragB[nii], 
                        Accum[mii * (wn / wmma_n) + nii]
                    );
                }
            }
        }

        __syncthreads();
        if (ko + 5 < K / bk) {
            loadSmemA(SA2, A, M, K, ko + 5);
            loadSmemB(SB2, B, N, K, ko + 5);
        }
        for (int ki = 0; ki < bk / wk; ki += 1) {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA3, ki);
            loadFragB(FragB, SB3, ki);
            for (int mii = 0; mii < wm / wmma_m; mii += 1) {
                for (int nii = 0; nii < wn / wmma_n; nii += 1) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(
                        Accum[mii * (wn / wmma_n) + nii], 
                        FragA[mii], FragB[nii], 
                        Accum[mii * (wn / wmma_n) + nii]
                    );
                }
            }
        }

        __syncthreads();
        if (ko + 6 < K / bk) {
            loadSmemA(SA3, A, M, K, ko + 6);
            loadSmemB(SB3, B, N, K, ko + 6);
        }
        for (int ki = 0; ki < bk / wk; ki += 1) {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA4, ki);
            loadFragB(FragB, SB4, ki);
            for (int mii = 0; mii < wm / wmma_m; mii += 1) {
                for (int nii = 0; nii < wn / wmma_n; nii += 1) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(
                        Accum[mii * (wn / wmma_n) + nii], 
                        FragA[mii], FragB[nii], 
                        Accum[mii * (wn / wmma_n) + nii]
                    );
                }
            }
        }
    }
    
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}