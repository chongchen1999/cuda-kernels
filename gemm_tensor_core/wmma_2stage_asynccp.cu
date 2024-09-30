// wmma + pipeline

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

using namespace nvcuda;

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void loadSmemA(
    half *smem, half *A, 
    const int M, const int K, const int block_iter
) {
    // load 128 * 32
    const int by = blockIdx.y;
    const int lane_id = threadIdx.x;
    const int warp_x = threadIdx.y;
    const int warp_y = threadIdx.z;
    constexpr int threads = 32 * warps_m * warps_n;

    // layout: [8, 2, 16, 16]
    constexpr int rows_o_block = bm / wm;
    constexpr int cols_o_block = bk / wk;
    constexpr int rows_i_block = wmma_m;
    constexpr int cols_i_block = wmma_k;

    constexpr int halfs_per_thread = 128 / 16; // 128-bit per thread, aka 8 half per thread
    constexpr int halfs_per_warp = 32 * halfs_per_thread; // 32 threads, 128-bit per thread, 16 bit per half
    constexpr int warp_iters = bm * bk / (halfs_per_thread * threads);
    constexpr int rows_per_warp = halfs_per_warp / bk;
    constexpr int threads_per_row = bk / halfs_per_thread;

    const int tid = (warp_y * 32 * cols_o_block) + (warp_x * 32) + lane_id;

    #pragma unroll
    for (int i = 0; i < warp_iters; ++i) {
        const int row = (i * rows_per_warp) + (tid / threads_per_row); // 1 thread load 128-bit, 4 threads per row
        const int col = (lane_id / threads_per_row) * halfs_per_thread; // 128-bit per thread, aka 8 half per thread

        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        const int row_o = row / wmma_m;
        const int col_o = col / wmma_n;
        const int row_i = row % wmma_m;
        const int col_i = col % wmma_n;
        
        void *ptr = reinterpret_cast<void *>(
            smem + (row_o * cols_o_block * rows_i_block * cols_i_block) + 
            (col_o * rows_i_block * cols_i_block) + (row_i * cols_i_block) + col_i
        );
        uint32_t smem_ptr;

        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr)
        );

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n"
            :
            : "r"(smem_ptr), "l"(&A[(by * bm + row) * K + (block_iter * bk + col)]), "n"(16)
        );
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void loadSmemB(
    half *smem, half *B, 
    const int N, const int K, const int block_iter
) {
    // load 128 * 32
    const int bx = blockIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_x = threadIdx.y;
    const int warp_y = threadIdx.z;
    constexpr int threads = 32 * warps_m * warps_n;

    // layout: [8, 2, 16, 16]
    constexpr int rows_o_block = bm / wm;
    constexpr int cols_o_block = bk / wk;
    constexpr int rows_i_block = wmma_m;
    constexpr int cols_i_block = wmma_k;

    constexpr int halfs_per_thread = 128 / 16; // 128-bit per thread, aka 8 half per thread
    constexpr int halfs_per_warp = 32 * halfs_per_thread; // 32 threads, 128-bit per thread, 16 bit per half
    constexpr int warp_iters = bm * bk / (halfs_per_thread * threads);
    constexpr int rows_per_warp = halfs_per_warp / bk;
    constexpr int threads_per_row = bk / halfs_per_thread;

    const int tid = (warp_y * 32 * cols_o_block) + (warp_x * 32) + lane_id;

    #pragma unroll
    for (int i = 0; i < warp_iters; ++i) {
        const int row = (i * rows_per_warp) + (tid / threads_per_row); // 1 thread load 128-bit, 4 threads per row
        const int col = (lane_id / threads_per_row) * halfs_per_thread; // 128-bit per thread, aka 8 half per thread

        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        const int row_o = row / wmma_m;
        const int col_o = col / wmma_n;
        const int row_i = row % wmma_m;
        const int col_i = col % wmma_n;

        void *ptr = reinterpret_cast<void *>(
            smem + (row_o * cols_o_block * rows_i_block * cols_i_block) + 
            (col_o * rows_i_block * cols_i_block) + (row_i * cols_i_block) + col_i
        );
        uint32_t smem_ptr;

        asm(
            "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr)
        );

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n" 
            :
            : "r"(smem_ptr), "l"(&B[(bx * bn + row) * K + (block_iter * bk + col)]), "n"(16)
        );
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void storeSmemC(
    half *C, float *smem, 
    const int M, const int N
) {
    // load 128 * 128
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int lane_id = threadIdx.x;
    const int warp_x = threadIdx.y;
    const int warp_y = threadIdx.z;
    constexpr int threads = 32 * warps_m * warps_n;

    // layout: [8, 8, 16, 16]
    constexpr int rows_o_block = bm / wm;
    constexpr int cols_o_block = bk / wk;
    constexpr int rows_i_block = wmma_m;
    constexpr int cols_i_block = wmma_k;

    constexpr int halfs_per_thread = 128 / 16; // 128-bit per thread, aka 8 half per thread
    constexpr int halfs_per_warp = 32 * halfs_per_thread; // 32 threads, 128-bit per thread, 16 bit per half
    constexpr int warp_iters = bm * bk / (halfs_per_thread * threads);
    constexpr int rows_per_warp = halfs_per_warp / bk;
    constexpr int threads_per_row = bk / halfs_per_thread;

    const int tid = (warp_y * 32 * cols_o_block) + (warp_x * 32) + lane_id;

    #pragma unroll
    for (int i = 0; i < bm; ++i) {
        const int row = i;
        const int col = tid;

        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        const int row_o = row / wmma_m;
        const int col_o = col / wmma_n;
        const int row_i = row % wmma_m;
        const int col_i = col % wmma_n;

        C[(by * bm + row) * N + bx * bm + col] = static_cast<half>(
            *(smem + (row_o * rows_o_block * rows_i_block * cols_i_block) + 
            (col_o * rows_i_block * cols_i_block) + (row_i * cols_i_block) + col_i)
        );
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void loadFragA(
    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, wmma::row_major> *frag, 
    half *smem, const int warp_iter
) {
    // load 64x16
    const int warp_y = threadIdx.z;
    constexpr int frags_m = wm / wmma_m;

    #pragma unroll
    for (int i = 0; i < frags_m; ++i) {
        const int row = (warp_y * wm) + (i * wmma_m);
        const int col = warp_iter * wk;

        // layout: [8, 2, 16, 16]
        constexpr int rows_o_block = bm / wm;
        constexpr int cols_o_block = bk / wk;
        constexpr int rows_i_block = wmma_m;
        constexpr int cols_i_block = wmma_k;

        const int row_o = row / rows_i_block;
        const int col_o = col / cols_i_block;

        wmma::load_matrix_sync(
            frag[i], 
            smem + (row_o * cols_o_block * rows_i_block * cols_i_block) + (col_o * rows_i_block * cols_i_block), 
            wmma_k
        );
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void loadFragB(
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, wmma::col_major> *frag, 
    half *smem, const int warp_iter
) {
    // load 64x16
    const int warp_x = threadIdx.y;
    constexpr int frags_n = (wn / wmma_n);

    #pragma unroll
    for (int i = 0; i < frags_n; ++i) {
        const int row = (warp_x * wn) + (i * wmma_n);
        const int col = warp_iter * wk;

        // layout: [8, 2, 16, 16]
        constexpr int rows_o_block = bn / wn;
        constexpr int cols_o_block = bk / wk;
        constexpr int rows_i_block = wmma_n;
        constexpr int cols_i_block = wmma_k;

        const int row_o = row / rows_i_block;
        const int col_o = col / cols_i_block;
        
        wmma::load_matrix_sync(
            frag[i], 
            smem + (row_o * cols_o_block * rows_i_block * cols_i_block) + (col_o * rows_i_block * cols_i_block), 
            wmma_k
        );
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void storeAccum(
    float *SC, 
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float> *frag
) {
    // store 64x64
    const int warp_x = threadIdx.y;
    const int warp_y = threadIdx.z;
    constexpr int frags_m = wm / wmma_m;
    constexpr int frags_n = wn / wmma_n;

    #pragma unroll
    for (int i = 0; i < frags_m; ++i) {
        #pragma unroll
        for (int j = 0; j < frags_n; ++j) {
            const int row = (warp_y * wm) + (i * wmma_m);
            const int col = (warp_x * wn) + (j * wmma_n);

            // laoyut: [8, 8, 16, 16]
            constexpr int rows_o_block = bm / wm;
            constexpr int cols_o_block = bn / wn;
            constexpr int rows_i_block = wmma_m;
            constexpr int cols_i_block = wmma_n;

            const int row_o = row / rows_i_block;
            const int col_o = col / cols_i_block;
            
            wmma::store_matrix_sync(
                SC + (row_o * cols_o_block * rows_i_block * cols_i_block) + (col_o * rows_i_block * cols_i_block), 
                frag[i * frags_n + j], wmma_n, 
                wmma::mem_row_major
            );
        }
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void warpMMA(
    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, wmma::row_major> *frag_a, 
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, wmma::col_major> *frag_b, 
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float> *accum,
    half *SA, half *SB
) {
    constexpr int frags_m = wm / wmma_m;
    constexpr int frags_n = wn / wmma_n;
    constexpr int warp_iters = bk / wk;

    #pragma unroll
    for (int k = 0; k < warp_iters; ++k) {
        // 64x64x16 mma for each warp
        loadFragA(frag_a, SA, k);
        loadFragB(frag_b, SB, k);

        #pragma unroll
        for (int i = 0; i < frags_m; ++i) {
            #pragma unroll
            for (int j = 0; j < frags_n; ++j) {
                // 16x16x16 for each wmma
                wmma::mma_sync(
                    accum[i * frags_n + j], 
                    frag_a[i], frag_b[j], 
                    accum[i * frags_n + j]
                );
            }
        }
    }
}

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__device__ __forceinline__ void loadSmemAndCommit(
    half *SA, half *SB, 
    half *A, half *B, 
    const int block_iter, 
    const int M, const int N, const int K
) {
    loadSmemA(SA, A, M, K, block_iter);
    loadSmemB(SB, B, N, K, block_iter);
    asm volatile("cp.async.commit_group;\n" ::);
}

/*
A is row-major
B is col-major
128 threads [x, y, z] = [32, 2, 2]
threadblock mma: 128x128x32
warp mma: 64x64x16
*/

template <
    int bm = 128, int bn = 128, int bk = 32, 
    int wm = 64, int wn = 64, int wk = 16,
    int wmma_m = 16, int wmma_n = 16, int wmma_k = 16,
    int warps_m = 2, int warps_n = 2
>
__global__ void matmul(
    half *A, half *B, half *C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
) {
    extern __shared__ char shared_storage[];
    half *SA1 = reinterpret_cast<half *>(shared_storage);
    half *SA2 = SA1 + bm * bk;
    half *SB1 = SA2 + bm * bk;
    half *SB2 = SB1 + bn * bk;
    float *SC = reinterpret_cast<float *>(shared_storage);

    constexpr int frags_m = wm / wmma_m;
    constexpr int frags_n = wn / wmma_n;
    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, wmma::row_major> frag_a[frags_m];
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, wmma::col_major> frag_b[frags_n];
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float> accum[frags_m * frags_n];

    for (int i = 0; i < frags_m * frags_n; ++i) {
        wmma::fill_fragment(accum[i], 0.0);
    }
    
    // prologue
    loadSmemAndCommit(SA1, SB1, A, B, 0, M, N, K);

    const int block_iters = K / bk;

    #pragma unroll
    for (int ko = 0; ko + 2 < block_iters; ko += 2) {
        loadSmemAndCommit(SA2, SB2, A, B, ko + 1, M, N, K);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        warpMMA(frag_a, frag_b, accum, SA1, SB1);

        loadSmemAndCommit(SA1, SB1, A, B, ko + 2, M, N, K);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        warpMMA(frag_a, frag_b, accum, SA2, SB2);
    }

    {
        int ko = (block_iters / 2 - 1) * 2;

        if (ko < block_iters) {
            warpMMA(frag_a, frag_b, accum, SA1, SB1);
        }
        if (ko + 1 < block_iters) {
            warpMMA(frag_a, frag_b, accum, SA2, SB2);
        }
    }

    storeAccum(SC, accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}

template __global__ void matmul<128, 128, 32, 64, 64, 16, 16, 16, 16, 2, 2>(
    half *A, half *B, half *C, 
    const int M, const int N, const int K, 
    const float alpha, const float beta
);