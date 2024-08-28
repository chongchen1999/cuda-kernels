#include "includes/sgemm.cuh"

template <
    int block_tile_m = 128, 
    int block_tile_n = 128, 
    int block_tile_k = 8, 
    int thread_tile_m = 8, 
    int thread_tile_n = 8
>
__global__ void sgemmKernel(
    float *device_C, float *device_A, float *device_B, 
    int M, int N, int K, 
    float alpha, float beta
) {
    const int block_num = gridDim.x;
    const int thread_num = blockDim.x;
    const int tid = threadIdx.x;
    
    const int block_tile_per_row = (N + block_tile_n - 1) / block_tile_n;
    const int block_tile_per_col = (M + block_tile_m - 1) / block_tile_m;

    const int block_tile_x = block_num % block_tile_per_row;
    const int block_tile_y = block_num / block_tile_per_row;

    const int block_tile_base_offset_a = K * block_tile_y * block_tile_m;
    const int block_tile_base_offset_b = block_tile_x * block_tile_n;
    const int block_tile_stride_a = block_tile_k;
    const int block_tile_stride_b = block_tike_k * N;

    const int cur_thread_row_within_block_tile_a = (tid << 2) / block_tile_m;
    const int cur_thread_col_within_block_tile_a = (tid << 2) % block_tile_m;
    const int cur_thread_row_within_block_tile_b = (tid << 2) / block_tile_n;
    const int cur_thread_col_within_block_tile_b = (tid << 2) % block_tile_n;
    const int thread_stride = thread_num << 2;

    __shared__ float shared_a[block_tile_k * block_tile_m];
    __shared__ float shared_b[block_tile_k * block_tile_n];

    float cum_c[thread_tile_m][thread_tile_n] = {0.f};
    float4 buffer_global_reg_a[(block_tile_m * block_tile_k + thread_stride - 1) / thread_stride];
    float4 buffer_global_reg_b[(block_tile_n * block_tile_k + thread_stride - 1) / thread_stride];
    float buffer_shared_reg_a[thread_tile_m];
    float buffer_shared_reg_b[thread_tile_n];

    const int thread_tile_per_row = (block_tile_n + thread_tile_n - 1) / thread_tile_n;
    const int thread_tile_per_col = (block_tile_m + thread_tile_m - 1) / thread_tile_m;
    const int thread_tile_x = thread_num % thread_tile_per_row;
    const int thread_tile_y = thread_num / thread_tile_per_row;

    int idx_block_tile = 0;
    for (int block_tile_k_iter = 0; block_tile_k_iter < K; block_tile_k_iter += block_tile_k) {
        load_global_to_buffer_reg_a(
            buffer_global_reg_a,
            device_A,
            block_tile_base_offset_a + idx_block_tile * block_tile_stride_a,
            cur_thread_row_within_block_tile_a,
            cur_thread_col_within_block_tile_a,
            M, N, K,
            block_tile_m, block_tile_n, block_tile_k
        );
        load_global_to_buffer_reg_b(
            buffer_global_reg_b,
            device_B,
            block_tile_base_offset_b + idx_block_tile * block_tile_stride_b,
            cur_thread_row_within_block_tile_b,
            cur_thread_col_within_block_tile_b
            M, N, K,
            block_tile_m, block_tile_n, block_tile_k
        );
        load_buffer_reg_to_shared(
            shared_a,
            cur_thread_row_within_block_tile_a,
            cur_thread_col_within_block_tile_a,
            buffer_global_reg_a,
            M, N, K,
            block_tile_m, block_tile_n, block_tile_k
        );
        load_buffer_reg_to_shared(
            shared_b,
            cur_thread_row_within_block_tile_b,
            cur_thread_col_within_block_tile_b,
            buffer_global_reg_b,
            M, N, K,
            block_tile_m, block_tile_n, block_tile_k
        );
        __syncthreads();


        for (int thread_tile_k = 0; thread_tile_k < block_tile_k; ++thread_tile_k) {
            load_shared_to_reg(
                buffer_shared_reg_a,
                shared_a,
                thread_tile_k,
                thread_tile_y,
                M, N, K,
                block_tile_m, block_tile_n, block_tile_k
                thread_tile_m, thread_tile_n
            );
            load_shared_to_reg(
                buffer_shared_reg_b,
                shared_b,
                thread_tile_k,
                thread_tile_x,
                M, N, K,
                block_tile_m, block_tile_n, block_tile_k
                thread_tile_m, thread_tile_n
            );

            for (int i = 0; i < thread_tile_m; ++i) {
                for (int j = 0; j < thread_tile_n; ++j) {
                    cum_c[i][j] += buffer_shared_reg_a[i] * buffer_shared_reg_b[j];
                }
            }
        }
        ++idx_block_tile;
    }

    for (int i = 0; i < thread_tile_m; ++i) {
        for (int j = 0; j < thread_tile_n; ++j) {
            device_C[block_tile_base_offset_c + i * block_tile_stride_c + j] += cum_c[i][j];
        }
    }

}