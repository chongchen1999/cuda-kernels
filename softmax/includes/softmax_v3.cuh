#pragma once

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <cmath>
#include <functional>

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
    
    template <typename T, int nums_per_thread>
    __global__ void blockBasedSoftmax(const T *input, T *output, const int M, const int N) {
        // printf("get in!\n");
        const int tid = threadIdx.x;
        const int vec_tid = tid << 2;
        const int block_stride = blockDim.x << 2;
        const int grid_stride = gridDim.x;

        __shared__ T block_max;
        __shared__ T block_sum;
        float4 data[nums_per_thread];

        #pragma unroll
        for (int row = blockIdx.x; row < M; row += grid_stride) {
            T max = MaxOp<T>::identity;

            #pragma unroll
            for (int col = vec_tid, idx = 0; col < N; col += block_stride, ++idx) {
                const float4 ival = *reinterpret_cast<const float4 *>(input + row * N + col);
                max = MaxOp<T>()(max, ival.x);
                max = MaxOp<T>()(max, ival.y);
                max = MaxOp<T>()(max, ival.z);
                max = MaxOp<T>()(max, ival.w);
                data[idx] = ival;
            }
            blockReduce<T, MaxOp>(max);
            if (tid == 0) {
                block_max = max;
            }
            __syncthreads();
            max = block_max;

            T sum = SumOp<T>::identity;
            #pragma unroll
            for (int i = vec_tid, idx = 0; i < N; i += block_stride, ++idx) {
                float4 &ival = data[idx];
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
            for (int col = vec_tid, idx = 0; col < N; col += block_stride, ++idx) {
                float4 &oval = *reinterpret_cast<float4 *>(output + row * N + col);
                const float4 &sval = data[idx];
                oval.x = sval.x / sum;
                oval.y = sval.y / sum;
                oval.z = sval.z / sum;
                oval.w = sval.w / sum;
            }
        }
    }

    template <typename T>
    void launchSoftmax(const T *input, T *output, const int M, const int N, const int times = 1) {
        const int vec_N = (N + 3) >> 2;
        const int block_size = std::min(256, vec_N);
        const int grid_size = std::min(1024 , M);
        // const int grid_size = 100;

        std::cout << "Softmax: block_size = " << block_size << ", grid_size = " << grid_size << std::endl;

        dim3 block_shape(block_size);
        dim3 grid_shape(grid_size);

        using SoftmaxKernel = std::function<void(const T*, T*, const int, const int, dim3, dim3)>;
        std::map<int, SoftmaxKernel> kernel_map = {
            {1, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 1><<<grid, block>>>(input, output, M, N);
            }},
            {2, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 2><<<grid, block>>>(input, output, M, N);
            }},
            {4, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 4><<<grid, block>>>(input, output, M, N);
            }},
            {8, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 8><<<grid, block>>>(input, output, M, N);
            }},
            {16, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 16><<<grid, block>>>(input, output, M, N);
            }},
            {32, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 32><<<grid, block>>>(input, output, M, N);
            }},
            {64, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 64><<<grid, block>>>(input, output, M, N);
            }},
            {128, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 128><<<grid, block>>>(input, output, M, N);
            }},
            {256, [](const T* input, T* output, int M, int N, dim3 grid, dim3 block) {
                blockBasedSoftmax<T, 256><<<grid, block>>>(input, output, M, N);
            }}
        };

        float elapse = .0f;

        const int temp = (vec_N + block_size - 1) / block_size;
        const int nums_per_thread = 1 << static_cast<int>(std::ceil(std::log2(temp)));
        printf("%d\n", nums_per_thread);
        auto softmax_kernel = kernel_map.find(nums_per_thread);
        if (softmax_kernel == kernel_map.end()) {
            printf("error!");
            return;
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < times; ++ i) {
            softmax_kernel->second(input, output, M, N, grid_shape, block_shape);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapse, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Softmax: %f ms\n", elapse / times);
    }
}