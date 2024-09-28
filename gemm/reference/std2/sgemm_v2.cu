#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int iterations = 1000;

void init_matrix(float *A, int M, int NN, float value = 1.0) {
    for (int i = 0; i < M * NN; ++i) {
        A[i] = value * (float)rand() / RAND_MAX;
    }
}

float sgemm_cublas(
    float *device_A, float *device_B, 
    float *device_cublas, float *host_cublas, 
    int MM, int KK, int NN
) {
    float alpha = 1.0;
    float beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < iterations; ++i) {
        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            NN, MM, KK, &alpha, 
            device_B, NN, 
            device_A, KK, 
            &beta, 
            device_cublas, NN
        );
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float millionseconds = 0.0;
    cudaEventElapsedTime(&millionseconds, start, stop);

    millionseconds /= iterations;
    float gflops = (2.0 * MM * NN * KK) / (1 << 30) / (millionseconds / 1000.0);
    printf("cublas used time: %f ms\n", millionseconds);
    printf("cublas performance: %f GFLOPS\n", gflops);

    cudaMemcpy(host_cublas, device_cublas, MM * NN * sizeof(float), cudaMemcpyDeviceToHost);
    return millionseconds;
}

__device__ __forceinline__ void load_global_to_register(
    float *const &buffer, 
    const float * const &A, 
    const int &bm, const int &bn, 
    const int &M, const int &NN, 
    const int &start_row, const int &start_col, const int &stride
) {
    #pragma unroll
    for (int i = 0; i < bm; i += stride) {
        const int offset = i / stride;
        *(reinterpret_cast<float4 *>(buffer) + offset) = 
            *(reinterpret_cast<const float4 *>(A + (start_row + i) * NN + start_col));
    }
}

__device__ __forceinline__ void load_register_to_shared_a(
    const float *const &buffer, 
    float *const &shared, 
    const int &bm, const int &bn, 
    const int &M, const int &NN, 
    const int &start_row, const int &start_col, const int &stride
) {
    #pragma unroll
    for (int i = 0; i < bm; i += stride) {
        const int offset = i / stride;
        const float4 temp_reg = *(reinterpret_cast<const float4 *>(buffer) + offset);
        *(shared + (start_row + i) + bm * start_col) = temp_reg.x;
        *(shared + (start_row + i) + bm * (start_col + 1)) = temp_reg.y;
        *(shared + (start_row + i) + bm * (start_col + 2)) = temp_reg.z;
        *(shared + (start_row + i) + bm * (start_col + 3)) = temp_reg.w;
    }
}

__device__ __forceinline__ void load_register_to_shared_b(
    const float * const &buffer, 
    float * const &shared, 
    const int &bm, const int &bn, 
    const int &MM, const int &NN, 
    const int &start_row, const int &start_col, const int &stride
) {
    #pragma unroll
    for (int i = 0; i < bm; i += stride) {
        const int offset = i / stride;
        *(reinterpret_cast<float4 *>(shared + (start_row + i) * bn + start_col)) = 
            *(reinterpret_cast<const float4 *>(buffer) + offset);
    }
}

__device__ __forceinline__ void load_shared_to_register(
    float * const &reg, 
    const float * const &shared, 
    const int &bm, const int &bn, const int &len, 
    const int &offset_row, const int &offset_col
) {
    #pragma unroll
    for (int i = 0; i < len; i += 4) {
        *(reinterpret_cast<float4 *>(reg + i)) = 
            *(reinterpret_cast<const float4 *>(shared + offset_row * bn + (offset_col + i)));
    }
}

template <
    int block_tile_m, int block_tile_n, int block_tile_k, 
    int thread_tile_m, int thread_tile_n
>
__global__ void sgemm(
    float *device_A, float *device_B, float *device_C, 
    int MM, int KK, int NN, 
    float alpha, float beta
) {
    const int tid = threadIdx.x;
    const int number_of_thread_per_row = block_tile_n / thread_tile_n;
    const int number_of_thread_per_col = block_tile_m / thread_tile_m;
    const int number_of_threads = number_of_thread_per_row * number_of_thread_per_col;
    const int thread_x = threadIdx.x % number_of_thread_per_row * thread_tile_n;
    const int thread_y = threadIdx.x / number_of_thread_per_row * thread_tile_m;
    const int loads_per_thread_a = block_tile_m * block_tile_k / (number_of_threads << 2);
    const int loads_per_thread_b = block_tile_k * block_tile_n / (number_of_threads << 2);

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

    const float *A = device_A + blockIdx.y * block_tile_m * KK;
    const float *B = device_B + blockIdx.x * block_tile_n;
    float *C = device_C + blockIdx.y * block_tile_m * NN + blockIdx.x * block_tile_n;

    load_global_to_register(
        bufferReg_global_to_shared_a, A, 
        block_tile_m, block_tile_k, 
        MM, KK, 
        global_start_row_a, global_start_col_a, global_stride_a
    );
    load_global_to_register(
        bufferReg_global_to_shared_b, B, 
        block_tile_k, block_tile_n, 
        KK, NN, 
        global_start_row_b, global_start_col_b, global_stride_b
    );

    load_register_to_shared_a(
        bufferReg_global_to_shared_a, shared_a[0], 
        block_tile_m, block_tile_k, 
        MM, KK, 
        global_start_row_a, global_start_col_a, global_stride_a
    );
    load_register_to_shared_b(
        bufferReg_global_to_shared_b, shared_b[0], 
        block_tile_k, block_tile_n, 
        KK, NN, 
        global_start_row_b, global_start_col_b, global_stride_b
    );

    __syncthreads();

    load_shared_to_register(shared_to_reg_a[0], shared_a[0], block_tile_k, block_tile_m, thread_tile_m, 0, thread_y);
    load_shared_to_register(shared_to_reg_b[0], shared_b[0], block_tile_k, block_tile_n, thread_tile_n, 0, thread_x);

    int load_index_shared = 0;
    #pragma unroll
    for (int k = 0; k < KK; k += block_tile_k) {
        load_index_shared ^= 1;
        int next_k = k + block_tile_k;
        if (next_k < KK) {
            load_global_to_register(
                bufferReg_global_to_shared_a, A, 
                block_tile_m, block_tile_k, 
                MM, KK, 
                global_start_row_a, global_start_col_a + next_k, global_stride_a
            );
            load_global_to_register(
                bufferReg_global_to_shared_b, B, 
                block_tile_k, block_tile_n, 
                KK, NN, 
                global_start_row_b + next_k, global_start_col_b, global_stride_b
            );
        }

        int load_index_reg = 0;
        #pragma unroll
        for (int s = 0; s < block_tile_k; ++s) {
            load_index_reg ^= 1;
            if (s + 1 < block_tile_k) {
                load_shared_to_register(
                    shared_to_reg_a[load_index_reg], shared_a[load_index_shared ^ 1], 
                    block_tile_k, block_tile_m, thread_tile_m, 
                    s + 1, thread_y
                );
                load_shared_to_register(
                    shared_to_reg_b[load_index_reg], shared_b[load_index_shared ^ 1],
                    block_tile_k, block_tile_n, thread_tile_n, 
                    s + 1, thread_x
                );
            }

            #pragma unroll
            for (int i = 0; i < thread_tile_m; ++i) {
                #pragma unroll
                for (int j = 0; j < thread_tile_n; ++j) {
                    accum[i][j] += shared_to_reg_a[load_index_reg ^ 1][i] * shared_to_reg_b[load_index_reg ^ 1][j];
                }
            }
        }
        
        if (next_k < KK) {
            load_register_to_shared_a(
                bufferReg_global_to_shared_a, shared_a[load_index_shared], 
                block_tile_m, block_tile_k, 
                MM, KK, 
                global_start_row_a, global_start_col_a, global_stride_a
            );
            load_register_to_shared_b(
                bufferReg_global_to_shared_b, shared_b[load_index_shared], 
                block_tile_k, block_tile_n, 
                KK, NN, 
                global_start_row_b, global_start_col_b, global_stride_b
            );

            __syncthreads();

            load_shared_to_register(
                shared_to_reg_a[0], shared_a[load_index_shared], 
                block_tile_k, block_tile_m, thread_tile_m, 
                0, thread_y
            );
            load_shared_to_register(
                shared_to_reg_b[0], shared_b[load_index_shared], 
                block_tile_k, block_tile_n, thread_tile_n, 
                0, thread_x
            );
        }
    }

    #pragma unroll
    for (int i = 0; i < thread_tile_m; ++i) {
        #pragma unroll
        for (int j = 0; j < thread_tile_n; j += 4) {
            float4 &accum_ref = *reinterpret_cast<float4 *>(&accum[i][j]);
            float4 &C_ref = *reinterpret_cast<float4 *>(C + (thread_y + i) * NN + (thread_x + j));
            accum_ref.x = alpha * accum_ref.x + beta * C_ref.x;
            accum_ref.y = alpha * accum_ref.y + beta * C_ref.y;
            accum_ref.z = alpha * accum_ref.z + beta * C_ref.z;
            accum_ref.w = alpha * accum_ref.w + beta * C_ref.w;
            C_ref = accum_ref;
        }
    }
}

float sgemm_kernel(float *device_A, float *device_B, float *device_C, float *host_C, int MM, int KK, int NN) {
    float alpha = 1.0;
    float beta = 0.0;

    constexpr int block_tile_m = 128;
    constexpr int block_tile_n = 128;
    constexpr int block_tile_k = 8;
    constexpr int thread_tile_m = 8;
    constexpr int thread_tile_n = 8;
    constexpr int number_of_threads = 256;

    dim3 threads_per_block(number_of_threads);
    dim3 blocks_per_grid((MM - 1) / block_tile_m + 1, (NN - 1) / block_tile_n + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; ++i) {
        sgemm<block_tile_m, block_tile_n, block_tile_k, 
              thread_tile_m, thread_tile_n>
            <<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, MM, KK, NN, alpha, beta);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float millionseconds = 0.0;
    cudaEventElapsedTime(&millionseconds, start, stop);
    millionseconds /= iterations;
    float gflops = (2.0 * MM * NN * KK) / (1 << 30) / (millionseconds / 1000.0);
    printf("sgemm kernel used time: %f ms\n", millionseconds);
    printf("sgemm kernel performance: %f GFLOPS\n", gflops);

    cudaMemcpy(host_C, device_C, MM * NN * sizeof(float), cudaMemcpyDeviceToHost);
    return millionseconds;
}

bool check_reult(float *A, float *B, int size) {
    const auto eps = 1e-3;
    for (int i = 0; i < size; ++i) {
        auto x = A[i];
        auto y = B[i];
        if (fabs(x - y) / (fabs(x) + eps) > eps) {
            printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    float *host_A, *host_B, *host_C, *host_cublas;
    float *device_A, *device_B, *device_C, *device_cublas;

    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        return 1;
    }

    int MM = atoi(argv[1]);
    int KK = atoi(argv[2]);
    int NN = atoi(argv[3]);

    host_A = (float *)malloc(MM * KK * sizeof(float));
    host_B = (float *)malloc(KK * NN * sizeof(float));
    host_C = (float *)malloc(MM * NN * sizeof(float));
    host_cublas = (float *)malloc(MM * NN * sizeof(float));

    cudaMalloc((void **) &device_A, MM * KK * sizeof(float));
    cudaMalloc((void **) &device_B, KK * NN * sizeof(float));
    cudaMalloc((void **) &device_C, MM * NN * sizeof(float));
    cudaMalloc((void **) &device_cublas, MM * NN * sizeof(float));

    init_matrix(host_A, MM, KK);
    init_matrix(host_B, KK, NN);
    init_matrix(host_C, MM, NN, 0.0);
    init_matrix(host_cublas, MM, NN, 0.0);

    cudaMemcpy(device_A, host_A, MM * KK * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, KK * NN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, MM * NN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_cublas, host_C, MM * NN * sizeof(float), cudaMemcpyHostToDevice);

    auto kernel_time = sgemm_kernel(device_A, device_B, device_C, host_C, MM, KK, NN);
    auto cublas_time = sgemm_cublas(device_A, device_B, device_cublas, host_cublas, MM, KK, NN);
    printf("ratio = %.4f%%\n", cublas_time / kernel_time * 100.0);

    if (check_reult(host_C, host_cublas, MM * NN)) {
        printf("result is right\n");
    } else {
        printf("result is wrong\n");
    }

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    cudaFree(device_cublas);
    free(host_A);
    free(host_B);
    free(host_C);
    free(host_cublas);
    return 0;
}