#include <iostream>

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator() (const T &A, const T &B) const {
        return A + B;
    }
    static const T identity = static_cast<T>(0);
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
    const int lane_id = tid & 31;
    __shared__ T warp_sum[32];
    const int warp_num = (blockDim.x + 31) >> 5;

    warpReduce<T, Operator>(val);
    if (lane_id == 0) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < warp_num ? warp_sum[lane_id] : Operator<T>::itentity;
        warpReduce(val);
    }
}


template <typename T>
__global__ void deviceReduce(const T *A, const T *B, const int N, T *C) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    T val = Operator<T>::identity;

    #pragma unroll
    for (int i = gid; i < N; i += stride) {
        val += A[i] * B[i];
    }

    blockReduce<T, SumOp>(val);
    if (tid == 0) {
        atomicAdd(C, val);
    }
}

int main() {

    return 0;
}