#include <ctime>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../utils/includes/cpu_random.h"

template <typename T>
struct SumOp {
    static const T identity = static_cast<T>(0);
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T, template <typename> class ReductionOp = SumOp>
__device__ __forceinline__ void warpReduce(T &val) {
    #pragma unroll
    for (unsigned int mask = 16; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
}

template <typename T, template <typename> class ReductionOp = SumOp>
__device__ void blockReduce(T &val) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_num = (blockDim.x + 31) >> 5;

    __shared__ T warp_reduced[32];
    warpReduce<T, ReductionOp>(val);
    if (lane_id == 0) {
        warp_reduced[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < warp_num ? warp_reduced[lane_id] : ReductionOp<T>::identity;
        warpReduce<T, ReductionOp>(val);
    }
}

template <
    typename T,
    template <typename> class ReductionOp = SumOp
>
__global__ void DeviceReduce(T *data, int N, T *device_result) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;
    T block_reduced = static_cast<T>(0);

    #pragma unroll
    for (int i = gid; i < N; i += grid_stride) {
        block_reduced = ReductionOp<T>()(block_reduced, data[i]);
    }

    blockReduce<T, ReductionOp>(block_reduced);

    if (tid == 0) {
        atomicAdd(device_result, block_reduced);
    }
}

int main() {
    const int N = 1 << 25;
    const int iterations = 1000;

    std::srand(233u);

    float *host_data, *device_data;
    host_data = new float[N];
    cudaMalloc(reinterpret_cast<void **>(&device_data), sizeof(float) * N);

    float cpu_sum = 0.f;
    for (int i = 0; i < N; ++i) {
        float &x = host_data[i];
        // x = std::pow(2, -16);
        x = static_cast<float>(randomTools::fastUnsignedShort()) / 65536.f;
        cpu_sum += x;
    }
    std::cout << "cpu sum: " << cpu_sum << std::endl;
    cudaMemcpy(device_data, host_data, sizeof(float) * N, cudaMemcpyHostToDevice);

    constexpr size_t block_size = 512;
    constexpr size_t grid_size = std::min(2048lu, (N + block_size - 1) / block_size);
    dim3 grid_shape(grid_size);
    dim3 block_shape(block_size);

    float *host_gpu_sum, *device_gpu_sum;
    host_gpu_sum = new float(0);
    cudaMalloc(reinterpret_cast<void **>(&device_gpu_sum), sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpy(device_gpu_sum, host_gpu_sum, sizeof(float), cudaMemcpyHostToDevice);
        DeviceReduce<float><<<grid_shape, block_shape>>>(device_data, N, device_gpu_sum);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds / iterations);
    
    cudaMemcpy(host_gpu_sum, device_gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(4); // Set fixed-point notation and precision

    std::cout << "cpu result: " << cpu_sum << std::endl;
    std::cout << "gpu result: " << *host_gpu_sum << std::endl;

    delete[] host_data;
    cudaFree(device_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}