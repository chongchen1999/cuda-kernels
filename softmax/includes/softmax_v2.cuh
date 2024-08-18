#pragma once

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

// using a warp to process a row of the matrix
namespace warpBasedSoftmax {
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

    template <typename T, int size>
    __global__ void warpBasedSoftmax(T *input, T *output, int M, int N) {
        const int tid = threadIdx.x;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        const int warps_per_block = (blockDim.x + 31) >> 5; // aka rows per block

        const int start_row = warps_per_block * blockIdx.x + warp_id;
        
        T ival[128];
        #pragma unroll
        for (int row = start_row; row < M; row += warps_per_block * gridDim.x) {
            T max = MaxOp<T>::identity;
            #pragma unroll
            for (int col = lane_id << 2, idx = 0; col < N; col += 128, idx += 4) { // vectorized
                float4 temp = *reinterpret_cast<float4 *>(input + row * N + col);
                *reinterpret_cast<float4 *>(ival + idx) = temp;
                max = MaxOp<T>()(max, temp.x);
                max = MaxOp<T>()(max, temp.y);
                max = MaxOp<T>()(max, temp.z);
                max = MaxOp<T>()(max, temp.w);
            }
            warpReduce<T, MaxOp>(max);

            T sum = SumOp<T>::identity;
            #pragma unroll
            for (int col = lane_id << 2, idx = 0; col < N; col += 128, idx += 4) { // vectorized
                float4 &temp = *reinterpret_cast<float4 *>(ival + idx);
                temp.x = exp(temp.x - max);
                temp.y = exp(temp.y - max);
                temp.z = exp(temp.z - max);
                temp.w = exp(temp.w - max);
                sum = SumOp<T>()(sum, temp.x);
                sum = SumOp<T>()(sum, temp.y);
                sum = SumOp<T>()(sum, temp.z);
                sum = SumOp<T>()(sum, temp.w);
            }
            warpReduce<T, SumOp>(sum);

            #pragma unroll
            for (int col = lane_id << 2, idx = 0; col < N; col += 128, idx += 4) {
                float4 temp = *reinterpret_cast<float4 *>(ival + idx);
                float4 &oval = *reinterpret_cast<float4 *>(output + row * N + col);
                oval.x = temp.x / sum;
                oval.y = temp.y / sum;
                oval.z = temp.z / sum;
                oval.w = temp.w / sum;
            }
        }
    }

    template <typename T>
    void launchSoftmax(T *input, T *output, int M, int N, int times = 1) {
        const int block_size = 512;
        const int warps_per_block = (block_size + 31) >> 5;
        const int grid_size = std::min(2048, (M + warps_per_block - 1) / warps_per_block);
        dim3 block_shape(block_size);
        dim3 grid_shape(grid_size);

        float elapse = .0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < times; ++ i) {
            warpBasedSoftmax<T, 64><<<grid_shape, block_shape>>>(input, output, M, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapse, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Softmax: %f ms\n", elapse / times);
    }
}