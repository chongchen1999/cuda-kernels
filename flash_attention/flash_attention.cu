#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
    dim3 grid_dim(batch_size, head_num);  // batch_size x num_heads
    dim3 block_dim(tile_width_q);  // tile_width_q threads per block
*/

__global__ void forward_kernel(
    const float *Q, const float *K, const float *V, // [bs, num_heads, N, d]
    float *O, // [bs, num_heads, N]
    const int N, const int d,
    const int tile_num_q, const int tile_num_kv, 
    const int tile_width_q, const int tile_width_kv, 
    const float softmax_scale,
    float *exp_sum, float *val_max // [bs, num_heads, N]
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x; 
    const int head_idx = blockIdx.y;
    const int head_num = gridDim.y;

    // Offset into Q,K,V,O,exp_sum,val_max - different for each batch and head
    const int qkv_base_offset = (batch_idx * head_num * N * d) + (head_idx * N * d);
    const int lm_base_offset = (batch_idx * head_num * N) + (head_idx * N);  // offset for exp_sum and val_max

    // Define SRAM for Q,K,V,P
    extern __shared__ float sram[];
    const int tile_size_q = tile_width_q * d;
    const int tile_size_kv = tile_width_kv * d;
    float *Qi = sram;
    float *Kj = &sram[tile_size_q];
    float *Vj = &sram[tile_size_q + tile_size_kv];
    float *P = &sram[tile_size_q + 2 * tile_size_kv]; // P = QiKi^T

    for (int j = 0; j < tile_num_kv; j++) {
        // Load Kj, Vj to SRAM, make sure d can be divided by 4
        #pragma unroll
        for (int x = 0; x < d; x += 4) {
            const int head_offset = tid * d + x;
            const int kv_offset = qkv_base_offset + (tile_size_kv * j) + head_offset;
            *reinterpret_cast<float4 *>(Kj + head_offset) = *reinterpret_cast<const float4 *>(K + kv_offset);
            *reinterpret_cast<float4 *>(Vj + head_offset) = *reinterpret_cast<const float4 *>(V + kv_offset);
        }
        __syncthreads();

        for (int i = 0; i < tile_num_q; i++)  {
            // Load Qi to SRAM, exp_sum and val_max to registers
            for (int x = 0; x < d; x += 4) {
                const int head_offset = tid * d + x;
                *reinterpret_cast<float4 *>(Qi + head_offset) = 
                    *reinterpret_cast<const float4 *>(Q + qkv_base_offset + (tile_size_q * i) + head_offset);
            }

            const int lm_offset = lm_base_offset + (tile_width_kv * i) + tid;
            float row_m_max_prev = val_max[lm_offset];
            float row_exp_sum_prev = exp_sum[lm_offset];

            // P = QK^T, row_max = rowmax(P)
            float row_max = -INFINITY;
            for (int y = 0; y < tile_width_kv; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tid * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                P[(tile_width_kv * tid) + y] = sum;

                if (sum > row_max) {
                    row_max = sum;
                }
            }

            // P = exp(P - row_max), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < tile_width_q; y++) {
                float &p = P[(tile_width_q * tid) + y];
                p = __expf(p - row_max);
                row_l += p;
            }

            // Compute new val_max and exp_sum
            const float row_m_new = max(row_m_max_prev, row_max);
            const float row_l_new = __expf(row_m_max_prev - row_m_new) * row_exp_sum_prev + __expf(row_max - row_m_new) * row_l;

            // Write O, exp_sum, val_max to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < tile_width_q; y++) {
                    pv += P[(tile_width_q * tid) + y] * Vj[(y * d) + x];
                }
                const int O_offset = qkv_base_offset + (tile_size_q * i) + (tid * d) + x;
                O[O_offset] = (1.0f / row_l_new) * 
                    (row_exp_sum_prev * __expf(row_m_max_prev - row_m_new) * O[O_offset] + __expf(row_max - row_m_new) * pv);
            }
            val_max[lm_offset] = row_m_new;
            exp_sum[lm_offset] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine tile_width_q, tile_width_kv dynamically
    const int tile_width_q = 32; 
    const int tile_width_kv = 32;

    const int batch_size = Q.size(0); 
    const int head_num = Q.size(1);
    const int seq_len = Q.size(2); 
    const int head_size = Q.size(3);

    const int tile_num_q = ceil((float)seq_len / tile_width_q); 
    const int tile_num_kv = ceil((float)seq_len / tile_width_kv);
    const float softmax_scale = 1.0 / sqrt(head_size);

    // Initialize O, exp_sum, val_max to HBM
    auto O = torch::zeros_like(Q);
    auto exp_sum = torch::zeros({batch_size, head_num, seq_len});
    auto val_max = torch::full({batch_size, head_num, seq_len}, -INFINITY);
    torch::Device device(torch::kCUDA);
    exp_sum = exp_sum.to(device); 
    val_max = val_max.to(device);

    // Calculate SRAM size needed per block
    const int tile_size_q = tile_width_q * head_size;
    const int tile_size_kv = tile_width_kv * head_size;
    const int tile_size_s = tile_width_q * tile_width_kv; // s = qk^T
    const int sram_bytes = sizeof(float) * (tile_size_q + 2 * tile_size_kv + tile_size_s);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_bytes);

    dim3 grid_dim(batch_size, head_num);  // batch_size x num_heads
    dim3 block_dim(tile_width_q);  // tile_width_q threads per block

    forward_kernel<<<grid_dim, block_dim, sram_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
        O.data_ptr<float>(),
        seq_len, head_size, 
        tile_num_q, tile_num_kv, 
        tile_width_q, tile_width_kv, 
        softmax_scale,
        exp_sum.data_ptr<float>(), val_max.data_ptr<float>()
    );
    return O;
}