#include "includes/softmax_v1.h"
#include <algorithm>

// using a block to process a row of the matrix
namespace softmax {
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
    __device__ void warpReduce(T &val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
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
            val = tid < warp_num ? warp_reduced[land_id] : Operator<T>().identity();
            warpReduce<T, Operator>(val);
        }
    }
    
    template <typename T, int shared_size>
    __global__ void softmax_kernel(T *input, T *output, int M, int N) {
        const int tid = threadIdx.x;
        const int block_stride = blockDim.x << 2;
        const int grid_stride = gridDim.x;
    
        __shared__ T shared_val[shared_size];

        #pragma unroll
        for (int row = blockIdx.x; row < M; row += grid_stride) {
            T max = MaxOp<T>().identity();
            #pragma unroll
            for (int col = tid << 2; col < N; col += block_stride) {
                float4 ival = reinterpret_cast<float4 *>(input + row * N + col);
                max = MaxOp<T>()(max, ival.x);
                max = MaxOp<T>()(max, ival.y);
                max = MaxOp<T>()(max, ival.z);
                max = MaxOp<T>()(max, ival.w);
                shared_val[tid] = ival.x;
                shared_val[tid + 1] = ival.y;
                shared_val[tid + 2] = ival.z;
                shared_val[tid + 3] = ival.w;
            }
            blockReduce<T, MaxOp>(max);

            T sum = SumOp<T>().identity();
            #pragma unroll
            for (int i = 0; i < shared_size; i += 4) {
                float4 &ival = reinterpret_cast<float4 *>(shared_val + i);
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

            #pragma unroll
            for (int col = tid << 2, i = 0; col < N; col += block_stride, i += 4) {
                float4 &ival = reinterpret_cast<float4 *>(output + row * N + col);
                float4 sval = reinterpret_cast<float4 *>(shared_val + i);
                ival.x = sval.x / sum;
                ival.y = sval.y / sum;
                ival.z = sval.z / sum;
                ival.w = sval.w / sum;
            }
        }
    }

    template <typename T>
    void launchSoftmax(T *input, T *output, int M, int N) {
        const int vec_N = (N + 3) >> 2;
        const int block_size = std::min(512, vec_N);
        const int grid_size = std::min(2048, M);
        dim3 block_shape(block_size);
        dim3 grid_shape(grid_size);
        softmax_kernel<T, vec_N><<<grid_shape, block_shape>>>(input, output, M, N);
    }
}