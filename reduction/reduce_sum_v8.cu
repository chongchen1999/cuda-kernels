#include <cuda_runtime.h>
#include <iostream>
#include <cub/cub.cuh>  // CUB library header

int main() {
    int size = 1 << 25; // Size of the array
    int bytes = size * sizeof(float);
    int iterations = 1000;

    // Host memory
    float *host_data = new float[size];
    float host_result;

    // Initialize input data
    for (int i = 0; i < size; i++) {
        host_data[i] = 1.0f; // Example data
    }

    // Device memory
    float *device_data, *device_result;
    cudaMalloc(&device_data, bytes);
    cudaMalloc(&device_result, sizeof(float));

    // Copy data to device
    cudaMemcpy(device_data, host_data, bytes, cudaMemcpyHostToDevice);

    // CUB reduction
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_data, device_result, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, device_data, device_result, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds / iterations);

    // Copy result back to host
    cudaMemcpy(&host_result, device_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum: " << host_result << std::endl;

    // Free memory
    cudaFree(device_data);
    cudaFree(device_result);
    cudaFree(d_temp_storage);
    delete[] host_data;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}