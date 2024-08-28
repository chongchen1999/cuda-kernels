#include <iostream>
#include <cuda_runtime.h>

void queryGPUProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << device << " - " << prop.name << std::endl;
    std::cout << "Total Global Memory GBytes: " << prop.totalGlobalMem / (1 << 30) << std::endl;
    std::cout << "Shared Memory per Block KBytes: " << prop.sharedMemPerBlock / (1 << 10) << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Memory Clock Rate (GHz): " << prop.memoryClockRate / (1 << 20) << std::endl;
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "Clock Rate (KHz): " << prop.clockRate << std::endl;
    std::cout << "L2 Cache Size (MBytes): " << prop.l2CacheSize / (1 << 20) << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Maximum Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Maximum Block Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    int sharedMemPerSM;
    int regsPerSM;

    cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaDeviceGetAttribute(&regsPerSM, cudaDevAttrMaxRegistersPerMultiprocessor, device);

    std::cout << "Shared memory per SM: " << sharedMemPerSM / 1024 << " K bytes" << std::endl;
    std::cout << "Registers per SM: " << regsPerSM << std::endl;

     // Query cache configuration
    cudaFuncCache cacheConfig;
    cudaDeviceGetCacheConfig(&cacheConfig);

    std::cout << "Cache configuration: ";
    switch(cacheConfig) {
        case cudaFuncCachePreferNone:
            std::cout << "No preference" << std::endl;
            break;
        case cudaFuncCachePreferShared:
            std::cout << "Prefer shared memory over L1 cache" << std::endl;
            break;
        case cudaFuncCachePreferL1:
            std::cout << "Prefer L1 cache over shared memory" << std::endl;
            break;
        case cudaFuncCachePreferEqual:
            std::cout << "Prefer equal partitioning between L1 cache and shared memory" << std::endl;
            break;
        default:
            std::cout << "Unknown configuration" << std::endl;
            break;
    }

}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
        queryGPUProperties(device);
        std::cout << "-----------------------------------" << std::endl;
    }

    return 0;
}
