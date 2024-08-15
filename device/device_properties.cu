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
