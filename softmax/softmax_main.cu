#include "includes/softmax_v1.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include "../utils/includes/cpu_random.h"
#include "includes/softmax_cudnn.h"
#include "includes/softmax_cpu.h"
#include "../utils/includes/check_result.h"

int main() {
    const int M = 1 << 14;
    const int N = 2048;
    // const int N = 256;
    float *host_data, *device_data;
    host_data = static_cast<float *>(malloc(sizeof(float) * N * M));
    cudaMalloc(reinterpret_cast<void **>(&device_data), sizeof(float) * N * M);

    randomTools::randomFill<float>(host_data, M * N, 0.5f, 1.0f);
    cudaMemcpy(device_data, host_data, M * N * sizeof(float), cudaMemcpyHostToDevice);

    float *host_output, *device_output;
    host_output = static_cast<float *>(malloc(M * N * sizeof(float)));
    cudaMalloc(reinterpret_cast<void **>(&device_output), M * N * sizeof(float));

    float *host_cudnn_output, *cudnn_output;
    host_cudnn_output = static_cast<float *>(malloc(M * N * sizeof(float)));
    cudaMalloc(reinterpret_cast<void **>(&cudnn_output), M * N * sizeof(float));

    std::cout << "Start cuDNN!" << std::endl;
    cuDNN::launchSoftmax(device_data, cudnn_output, M, N);
    cudaMemcpy(host_cudnn_output, cudnn_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    softmax::launchSoftmax(device_data, device_output, M, N);
    cudaMemcpy(host_output, device_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float *cpu_result = static_cast<float *>(malloc(M * N * sizeof(float)));
    cpu_softmax::launchSoftmax(cpu_result, host_data, M, N);

    /*std::cout << "cpu result: " << std::endl;
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", cpu_result[i]);
    }
    puts("");

    std::cout << "my softmax result: " << std::endl;
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", host_output[i]);
    }
    puts("");

    std::cout << "cuDNN softmax result: " << std::endl;
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", host_cudnn_output[i]);
    }
    puts("");*/


    if (checkResult(host_output, host_cudnn_output, N)) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    cudaFree(device_data);
    cudaFree(device_output);
    cudaFree(cudnn_output);
    cudaFree(host_output);
    free(host_data);
    free(host_output);
    free(host_cudnn_output);
    free(cpu_result);

    return 0;
}