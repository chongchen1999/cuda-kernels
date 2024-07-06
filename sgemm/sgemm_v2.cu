#include <bits/stdc++.h>
#include <cublas_v2.h>

constexpr int iterations = 1000;

void init_matrix(float *A, int M, int N, float value = 1.0) {
    for (int i = 0; i < M * N; ++i) {
        A[i] = value * (float)rand() / RAND_MAX;
        A[i] = 1;
    }
}

float sgemm_cublas(float *device_A, float *device_B, float *device_cublas, float *host_cublas, int M, int K, int N) {
    float alpha = 1.0;
    float beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < iterations; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
                        device_B, N, device_A, K, &beta, device_cublas, N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float millionseconds = 0.0;
    cudaEventElapsedTime(&millionseconds, start, stop);

    millionseconds /= iterations;
    float gflops = (2.0 * M * N * K) / (1 << 30) / (millionseconds / 1000.0);
    printf("cublas used time: %f ms\n", millionseconds);
    printf("cublas performance: %f GFLOPS\n", gflops);

    cudaMemcpy(host_cublas, device_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    return millionseconds;
}

__device__ __forceinline__ void load_global_to_bufferReg(float * const &buffer, const float * const &A, 
    const int &bm, const int &bn, const int &M, const int &N, 
        const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        *((float4 *)buffer + offset) = *((float4 *)(A + (start_row + i) * N + start_col));
    }
}

__device__ __forceinline__ void load_bufferReg_to_shared_a(const float * const &buffer, float * const &shared, 
    const int &bm, const int &bn, const int &M, const int &N, 
        const int &start_row, const int &start_col, const int &stride) {
    for (int i = 0; i < bm; i += stride) {
        int offset = i / stride;
        const float4 temp_reg = *((float4 *)buffer + offset);
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
        *((float4 *)(shared + (start_row + i) * bn + start_col)) = *((float4 *)buffer + offset);
    }
}

__device__ __forceinline__ void load_shared_to_reg(float * const &reg, const float * const &shared, 
    const int &bm, const int &bn, const int &len, 
        const int &offset_row, const int &offset_col) {
    for (int i = 0; i < len; i += 4) {
        *((float4 *)(reg + i)) = *((float4 *)(shared + offset_row * bn + (offset_col + i)));
    }
}

template <int block_tile_m, int block_tile_n, int block_tile_k, 
    int thread_tile_m, int thread_tile_n>
__global__ void sgemm(float *device_A, float *device_B, float *device_C, 
    int M, int K, int N, float alpha, float beta) {
    const int tid = threadIdx.x;
    const int number_of_thread_per_row = block_tile_n / thread_tile_n;
    const int number_of_thread_per_col = block_tile_m / thread_tile_m;
    const int number_of_threads = number_of_thread_per_row * number_of_thread_per_col;
    const int thread_x = threadIdx.x % number_of_thread_per_row;
    const int thread_y = threadIdx.x / number_of_thread_per_row;
    const int loads_per_thread_a = block_tile_m * block_tile_k / (number_of_threads << 2);
    const int loads_per_thread_b = block_tile_k * block_tile_n / (number_of_threads << 2);
    const int shared_size_a = block_tile_m * block_tile_k;
    const int shared_size_b = block_tile_k * block_tile_n;

    int global_start_row_a = tid / (block_tile_k >> 2);
    int global_start_col_a = (tid % (block_tile_k >> 2)) << 2;
    int global_stride_a = (number_of_threads << 2) / block_tile_k;

    int global_start_row_b = tid / (block_tile_n >> 2);
    int global_start_col_b = (tid % (block_tile_n >> 2)) << 2;
    int global_stride_b = (number_of_threads << 2) / block_tile_n;

    __shared__ float shared_a[2][shared_size_a];
    __shared__ float shared_b[2][shared_size_b];
    float accum[thread_tile_m][thread_tile_n] = {0.0};

    float bufferReg_global_to_shared_a[loads_per_thread_a << 2] = {0.0};
    float bufferReg_global_to_shared_b[loads_per_thread_b << 2] = {0.0};

    float shared_to_reg_a[2][thread_tile_m];
    float shared_to_reg_b[2][thread_tile_n];

    const float *A = device_A + blockIdx.y * block_tile_m * K;
    const float *B = device_B + blockIdx.x * block_tile_n;
    float *C = device_C + blockIdx.y * block_tile_m * N + blockIdx.x * block_tile_n;

    load_global_to_bufferReg(bufferReg_global_to_shared_a, A, block_tile_m, block_tile_k, M, K, 
        global_start_row_a, global_start_col_a, global_stride_a);
    load_global_to_bufferReg(bufferReg_global_to_shared_b, B, block_tile_k, block_tile_n, K, N, 
        global_start_row_b, global_start_col_b, global_stride_b);

    load_bufferReg_to_shared_a(bufferReg_global_to_shared_a, shared_a[0], block_tile_m, block_tile_k, 
        M, K, global_start_row_a, global_start_col_a, global_stride_a);
    load_bufferReg_to_shared_b(bufferReg_global_to_shared_b, shared_b[0], block_tile_k, block_tile_n, 
        K, N, global_start_row_b, global_start_col_b, global_stride_b);

    __syncthreads();

    load_shared_to_reg(shared_to_reg_a[0], shared_a[0], block_tile_k, block_tile_m, 
        thread_tile_m, 0, thread_y * thread_tile_m);
    load_shared_to_reg(shared_to_reg_b[0], shared_b[0], block_tile_k, block_tile_n, 
        thread_tile_n, 0, thread_x * thread_tile_n);

    int load_index_shared = 0;
    for (int k = 0; k < K; k += block_tile_k) {
        load_index_shared ^= 1;
        int next_k = k + block_tile_k;
        if (next_k < K) {
            load_global_to_bufferReg(bufferReg_global_to_shared_a, A, block_tile_m, block_tile_k, M, K, 
                global_start_row_a, global_start_col_a + next_k, global_stride_a);
            load_global_to_bufferReg(bufferReg_global_to_shared_b, B, block_tile_k, block_tile_n, K, N,
                global_start_row_b + next_k, global_start_col_b, global_stride_b);
        }

        int load_index_reg = 0;
        for (int j = 0; j < block_tile_k; ++j) {
            load_index_reg ^= 1;
            if (j + 1 < block_tile_k) {
                load_shared_to_reg(shared_to_reg_a[load_index_reg], shared_a[load_index_shared ^ 1], 
                    block_tile_k, block_tile_m, thread_tile_m, j + 1, thread_y * thread_tile_m);
                load_shared_to_reg(shared_to_reg_b[load_index_reg], shared_b[load_index_shared ^ 1],
                    block_tile_k, block_tile_n, thread_tile_n, j + 1, thread_x * thread_tile_n);
            }
            
            for (int i = 0; i < thread_tile_m; ++i) {
                for (int j = 0; j < thread_tile_n; ++j) {
                    accum[i][j] += shared_to_reg_a[load_index_reg ^ 1][i] * shared_to_reg_b[load_index_reg ^ 1][j];
                }
            }
        }

        if (next_k < k) {
            load_bufferReg_to_shared_a(bufferReg_global_to_shared_a, shared_a[load_index_shared], 
                block_tile_m, block_tile_k, M, K, global_start_row_a, global_start_col_a, global_stride_a);
            load_bufferReg_to_shared_b(bufferReg_global_to_shared_b, shared_b[load_index_shared], 
                block_tile_k, block_tile_n, K, N, global_start_row_b, global_start_col_b, global_stride_b);

            __syncthreads();

            load_shared_to_reg(shared_to_reg_a[0], shared_a[load_index_shared], 
                block_tile_k, block_tile_m, thread_tile_m, 0, thread_y * thread_tile_m);
            load_shared_to_reg(shared_to_reg_b[0], shared_b[load_index_shared], 
                block_tile_k, block_tile_n, thread_tile_n, 0, thread_x * thread_tile_n);
        }
    }

    for (int i = 0; i < thread_tile_m; ++i) {
        for (int j = 0; j < thread_tile_n; ++j) {
            float &accum_ref = accum[i][j];
            float &C_ref = *(C + (thread_y * thread_tile_m + i) * N + (thread_x * thread_tile_n + j));
            accum_ref = alpha * accum_ref + beta * C_ref;
            C_ref = accum_ref;
        }
    }
}

float sgemm_kernel(float *device_A, float *device_B, float *device_C, float *host_C, int M, int K, int N) {
    float alpha = 1.0;
    float beta = 0.0;

    constexpr int block_tile_m = 128;
    constexpr int block_tile_n = 128;
    constexpr int block_tile_k = 8;
    constexpr int thread_tile_m = 8;
    constexpr int thread_tile_n = 8;
    constexpr int number_of_threads = 256;

    dim3 threads_per_block(number_of_threads);
    dim3 blocks_per_grid((M - 1) / block_tile_m + 1, (N - 1) / block_tile_n + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; ++i) {
        sgemm<block_tile_m, block_tile_n, block_tile_k, thread_tile_m, thread_tile_n>
            <<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, M, K, N, alpha, beta);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float millionseconds = 0.0;
    cudaEventElapsedTime(&millionseconds, start, stop);
    millionseconds /= iterations;
    float gflops = (2.0 * M * N * K) / (1 << 30) / (millionseconds / 1000.0);
    printf("sgemm kernel used time: %f ms\n", millionseconds);
    printf("sgemm kernel performance: %f GFLOPS\n", gflops);

    cudaMemcpy(host_C, device_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    return millionseconds;
}

bool check_reult(float *A, float *B, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > 1e-3) {
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

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    host_A = (float *)malloc(M * K * sizeof(float));
    host_B = (float *)malloc(K * N * sizeof(float));
    host_C = (float *)malloc(M * N * sizeof(float));
    host_cublas = (float *)malloc(M * N * sizeof(float));

    cudaMalloc((void **) &device_A, M * K * sizeof(float));
    cudaMalloc((void **) &device_B, K * N * sizeof(float));
    cudaMalloc((void **) &device_C, M * N * sizeof(float));
    cudaMalloc((void **) &device_cublas, M * N * sizeof(float));

    init_matrix(host_A, M, K);
    init_matrix(host_B, K, N);
    init_matrix(host_C, M, N, 0.0);
    init_matrix(host_cublas, M, N, 0.0);

    cudaMemcpy(device_A, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_cublas, host_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    auto cublas_time = sgemm_cublas(device_A, device_B, device_cublas, host_cublas, M, K, N);
    auto kernel_time = sgemm_kernel(device_A, device_B, device_C, host_C, M, K, N);
    printf("ratio = %.4f%%\n", cublas_time / kernel_time * 100.0);

    if (check_reult(host_C, host_cublas, M * N)) {
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