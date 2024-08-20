template <typename T>
struct SumOP {
    const static T identity = static_cast<T>(0);
    __device__ __forceinline__  T operator()(const T &a, const T &b) const {
        return a + b;
    }
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
    const int warp_num = blockDim.x >> 5;

    __shared__ T warp_reduced[32];
    warpReduce<T, Operator>(val);
    if (lane_id == 0) {
        warp_reduced[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = land_id < warp_num ? warp_reduced[lane_id] : Operator<T>::identity;
        warp_reduced<T, Operator>(val);
    }
}

template <typename T, template <typename> class Operator>
__device__ void deviceReduce(T *data, int N, T *result) {
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    T block_reduced = Operator<T>::identity;

    for (int i = tid; i < N; i += stride) {
        block_reduced = Operator<T>()(block_reduced, data[i]);
    }

    blockReduce<T, Operator>(block_reduced);
    if (threadIdx.x == 0) {
        atomicAdd(result, block_reduced);
    }
}