#include "includes/softmax_v1.cuh"
#include "includes/softmax_v2.cuh"
#include <iostream>
#include <memory>
#include <algorithm>
#include "../utils/includes/cpu_random.h"
#include "includes/softmax_cudnn.h"
#include "includes/softmax_cpu.h"
#include "../utils/includes/check_result.h"

int main() {
    const int M = 3000;
    // const int N = 13996;
    const int N = 8192;
    float *host_data, *device_data;
    host_data = static_cast<float *>(malloc(sizeof(float) * N * M));
    cudaMalloc(reinterpret_cast<void **>(&device_data), sizeof(float) * N * M);

    randomTools::randomFill<float>(host_data, M * N, 0.5f, 1.0f);
    cudaMemcpy(device_data, host_data, M * N * sizeof(float), cudaMemcpyHostToDevice);

    float *host_output_block_based, *host_output_warp_based, *device_output_block_based, *device_output_warp_based;
    host_output_block_based = static_cast<float *>(malloc(M * N * sizeof(float)));
    host_output_warp_based = static_cast<float *>(malloc(M * N * sizeof(float)));
    cudaMalloc(reinterpret_cast<void **>(&device_output_block_based), M * N * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&device_output_warp_based), M * N * sizeof(float));

    float *host_cudnn_output, *cudnn_output;
    host_cudnn_output = static_cast<float *>(malloc(M * N * sizeof(float)));
    cudaMalloc(reinterpret_cast<void **>(&cudnn_output), M * N * sizeof(float));

    int times = 1000;

    std::cout << "Start cuDNN!" << std::endl;
    cuDNN::launchSoftmax(device_data, cudnn_output, M, N, times);
    cudaMemcpy(host_cudnn_output, cudnn_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "cuDNN Done!" << std::endl << std::endl;

    std::cout << "Start block based softmax!" << std::endl;
    block_based_softmax::launchSoftmax(device_data, device_output_block_based, M, N, times);
    cudaMemcpy(host_output_block_based, device_output_block_based, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Block based softmax Done!" << std::endl << std::endl;

    /* std::cout << "Start warp based softmax!" << std::endl;
    warp_based_softmax::launchSoftmax(device_data, device_output_warp_based, M, N, times);
    cudaMemcpy(host_output_warp_based, device_output_warp_based, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Warp based softmax Done!" << std::endl << std::endl; */

    float *cpu_result = static_cast<float *>(malloc(M * N * sizeof(float)));
    cpu_softmax::launchSoftmax(cpu_result, host_data, M, N);

    /*std::cout << "cpu result: " << std::endl;
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", cpu_result[i]);
    }
    puts("");

    std::cout << "my softmax result: " << std::endl;
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", host_output_block_based[i]);
    }
    puts("");

    std::cout << "cuDNN softmax result: " << std::endl;
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", host_cudnn_output[i]);
    }
    puts("");*/


    if (checkResult(host_output_block_based, host_cudnn_output, N)) {
        std::cout << "Block Test passed!" << std::endl;
    } else {
        std::cout << "Block Test failed!" << std::endl;
    }

    if (checkResult(host_output_warp_based, host_cudnn_output, N)) {
        std::cout << "Warp Test passed!" << std::endl;
    } else {
        std::cout << "Warp Test failed!" << std::endl;
    }

    cudaFree(device_data);
    cudaFree(device_output_block_based);
    cudaFree(cudnn_output);
    cudaFree(host_output_block_based);
    cudaFree(device_output_warp_based);
    free(host_data);
    free(host_output_block_based);
    free(host_cudnn_output);
    free(cpu_result);
    free(host_output_warp_based);

    return 0;
}