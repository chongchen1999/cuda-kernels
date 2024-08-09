#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>

namespace cuDNN {
    void checkCUDNN(cudnnStatus_t status, int line) {
        if (status != CUDNN_STATUS_SUCCESS) {
            std::cerr 
                << "Error on line " << line << ": "
                << cudnnGetErrorString(status) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    #define CUDNN_CALL(func) checkCUDNN((func), __LINE__)

    template <typename T>
    void launchSoftmax(T *input, T *output, int M, int N, int times = 1) {
        cudnnHandle_t cudnn;
        CUDNN_CALL(cudnnCreate(&cudnn));

        cudnnTensorDescriptor_t input_desc, output_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));

        cudnnDataType_t data_type = std::is_same<T, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, data_type, M, N, 1, 1));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, data_type, M, N, 1, 1));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        float elapse = .0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int i = 0; i < times; ++i) {
            cudnnSoftmaxForward(
                cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, 
                &alpha, input_desc, input, 
                &beta, output_desc, output
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapse, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("Softmax: %f ms\n", elapse / times);

        CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
        CUDNN_CALL(cudnnDestroy(cudnn));
    }

}