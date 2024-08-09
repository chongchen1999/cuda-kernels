#include <iostream>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    checkCudaError(err, "Failed to get device count");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, dev);
        checkCudaError(err, "Failed to get device properties");

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Total Memory: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Clock Rate: " << (deviceProp.clockRate / 1000) << " MHz" << std::endl;
        std::cout << "  L2 Cache Size: " << (deviceProp.l2CacheSize / 1024) << " KB" << std::endl;
        std::cout << "  Max Threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Streaming Multiprocessors (SM) Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << std::endl;
    }

    return 0;
}