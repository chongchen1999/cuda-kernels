#pragma once

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// using a block to process a row of the matrix
namespace blockBasedSoftmax {
    template <typename T>
    struct SumOp {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const {
            return a + b;
        }
        static const T identity = static_cast<T>(0);
    };
    
    template <typename T>
    struct MaxOp {
        __device__ __forceinline__ T operator()(const T &a, const T &b) const {
            return a > b ? a : b;
        }
        static const T identity = static_cast<T>(-1e9);
    };

    template <typename T, template <typename> class Operator>
    __device__ __forceinline__ void warpReduce(T &val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = Operator<T>()(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
    }

    template <typename T, template <typename> class Operator>
    __device__ void blockReduce(T &val) {
        const int tid = threadIdx.x;
        const int warp_id = tid >> 5;
        const int land_id = tid & 31;
        const int warp_num = (blockDim.x + 31) >> 5;

        __shared__ T warp_reduced[32];
        warpReduce<T, Operator>(val);
        if (land_id == 0) {
            warp_reduced[warp_id] = val;
        }
        __syncthreads();

        if (warp_id == 0) {
            val = tid < warp_num ? warp_reduced[land_id] : Operator<T>::identity;
            warpReduce<T, Operator>(val);
        }
    }
    
    template <typename T>
    __global__ void blockBasedSoftmax(T *input, T *output, int M, int N) {
        // printf("get in!\n");
        const int tid = threadIdx.x;
        const int vec_tid = tid << 2;
        const int block_stride = blockDim.x << 2;
        const int grid_stride = gridDim.x;

        extern __shared__ T shared_val[];
        __shared__ T block_max;
        __shared__ T block_sum;

        #pragma unroll
        for (int row = blockIdx.x; row < M; row += grid_stride) {
            T max = MaxOp<T>::identity;
            #pragma unroll
            for (int col = vec_tid; col < N; col += block_stride) {
                float4 ival = *reinterpret_cast<float4 *>(input + row * N + col);
                max = MaxOp<T>()(max, ival.x);
                max = MaxOp<T>()(max, ival.y);
                max = MaxOp<T>()(max, ival.z);
                max = MaxOp<T>()(max, ival.w);
                shared_val[col] = ival.x;
                shared_val[col + 1] = ival.y;
                shared_val[col + 2] = ival.z;
                shared_val[col + 3] = ival.w;
            }
            blockReduce<T, MaxOp>(max);
            if (tid == 0) {
                block_max = max;
            }
            __syncthreads();
            max = block_max;

            T sum = SumOp<T>::identity;
            #pragma unroll
            for (int i = vec_tid; i < N; i += block_stride) {
                float4 &ival = *reinterpret_cast<float4 *>(shared_val + i);
                ival.x = exp(ival.x - max);
                ival.y = exp(ival.y - max);
                ival.z = exp(ival.z - max);
                ival.w = exp(ival.w - max);
                sum = SumOp<T>()(sum, ival.x);
                sum = SumOp<T>()(sum, ival.y);
                sum = SumOp<T>()(sum, ival.z);
                sum = SumOp<T>()(sum, ival.w);
            }
            blockReduce<T, SumOp>(sum);
            if (tid == 0) {
                block_sum = sum;
            }
            __syncthreads();
            sum = block_sum;

            #pragma unroll
            for (int col = vec_tid; col < N; col += block_stride) {
                float4 &oval = *reinterpret_cast<float4 *>(output + row * N + col);
                float4 sval = *reinterpret_cast<float4 *>(shared_val + col);
                oval.x = sval.x / sum;
                oval.y = sval.y / sum;
                oval.z = sval.z / sum;
                oval.w = sval.w / sum;
            }
        }
    }

    template <typename T>
    void launchSoftmax(T *input, T *output, int M, int N, int times = 1) {
        const int total_shared_bytes = 40 * 164 * 1024; // 40 SMs, 164KB per SM;
        const int vec_N = (N + 3) >> 2;
        const int block_size = std::min(512, vec_N);
        const int block_shared_bytes = (N + 34) * sizeof(T);
        const int grid_size = std::min(total_shared_bytes / block_shared_bytes , M);
        // const int grid_size = 100;

        std::cout << "Softmax: block_size = " << block_size << ", grid_size = " << grid_size << std::endl;

        dim3 block_shape(block_size);
        dim3 grid_shape(grid_size);

        float elapse = .0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < times; ++ i) {
            blockBasedSoftmax<T><<<grid_shape, block_shape, sizeof(T) * N>>>(input, output, M, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapse, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Softmax: %f ms\n", elapse / times);
    }
}