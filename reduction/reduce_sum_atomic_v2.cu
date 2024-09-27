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

template <
    typename T,
    template <typename> class ReductionOp = SumOp
>
__global__ void DeviceReduce(T *data, int N, T *device_result) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;

    __shared__ T block_sum;
    block_sum = ReductionOp<T>::identity;

    T thread_sum = ReductionOp<T>::identity;

    #pragma unroll
    for (int i = gid; i < N; i += grid_stride) {
        thread_sum += data[i];
    }
    
    atomicAdd(&block_sum, thread_sum);
    __syncthreads();

    if (tid == 0) {
        atomicAdd(device_result, block_sum);
    }
}

int main() {
    puts("Shared memory atomicAdd Test!");
    const int N = 8192;
    const int iterations = 1000;

    std::srand(233u);

    float *host_data, *device_data;
    host_data = new float[N];
    cudaMalloc(reinterpret_cast<void **>(&device_data), sizeof(float) * N);

    float cpu_sum = 0.f;
    randomTools::randomFill(host_data, N, 0.f, 1.f, 666);
    std::for_each(host_data, host_data + N, [&cpu_sum](float &val) { cpu_sum += val; });
    // std::cout << "cpu sum: " << cpu_sum << std::endl;
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
    printf("Time: %f ms\n", milliseconds);
    
    cudaMemcpy(host_gpu_sum, device_gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << std::fixed << std::setprecision(4); // Set fixed-point notation and precision

    std::cout << "cpu result: " << cpu_sum << std::endl;
    std::cout << "gpu result: " << *host_gpu_sum << std::endl;
    puts("");

    delete[] host_data;
    cudaFree(device_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}