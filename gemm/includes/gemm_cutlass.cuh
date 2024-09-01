#pragma once

#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <iostream>

namespace gemm_cutlass {
    void sgemm_cutlass(
        float *host_A, float *host_B, float *host_C, 
        int M, int K, int N,
        int iterations = 1000
    ) {
        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::Gemm<float,        // ElementA
                                                 cutlass::layout::RowMajor, // LayoutA
                                                 float,        // ElementB
                                                 cutlass::layout::RowMajor, // LayoutB
                                                 float,        // ElementC
                                                 cutlass::layout::RowMajor,  // LayoutC
                                                 float,        // ElementAccumulator
                                                 cutlass::arch::OpClassSimt,  // OperatorClass
                                                 cutlass::arch::Sm50>;  // ArchTag

        // Create a GEMM instance
        Gemm gemm_op;

        // Define the problem size
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        // Allocate device memory
        float *device_A, *device_B, *device_C;
        cudaMalloc((void**)&device_A, M * K * sizeof(float));
        cudaMalloc((void**)&device_B, K * N * sizeof(float));
        cudaMalloc((void**)&device_C, M * N * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(device_A, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_B, host_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

        // Define the arguments for the GEMM operation
        typename Gemm::Arguments arguments{
            problem_size,             // Problem size
            {device_A, K},            // Tensor-ref for A
            {device_B, N},            // Tensor-ref for B
            {device_C, N},            // Tensor-ref for C
            {device_C, N},            // Tensor-ref for D (output)
            {1.0f, 0.0f}              // Alpha, Beta
        };

        // Initialize the GEMM operation
        cutlass::Status status = gemm_op.initialize(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM initialization failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }

        // Warm-up run
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM execution failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
            return;
        }

        // Start timing
        float ker_time = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Execute the GEMM operation multiple times
        for (int i = 0; i < iterations; ++i) {
            status = gemm_op();
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "GEMM execution failed during iteration " << i << ": " << cutlass::cutlassGetStatusString(status) << std::endl;
                return;
            }
        }

        // End timing
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ker_time, start, stop);

        // Copy the result back to the host
        cudaMemcpy(host_C, device_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Print the elapsed time
        printf("kernel time: %.4f ms\n", ker_time / iterations);
    }
}