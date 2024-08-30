#include <iostream>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// Define the GEMM traits
using ColumnMajor = cutlass::layout::ColumnMajor;
using CutlassGemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;

int main() {
    const int M = 128;
    const int N = 128;
    const int K = 128;

    // Allocate matrices
    cutlass::HostTensor<float, ColumnMajor> A({M, K});
    cutlass::HostTensor<float, ColumnMajor> B({K, N});
    cutlass::HostTensor<float, ColumnMajor> C({M, N});

    // Initialize matrices (you can modify this part to set your own values)
    for (int i = 0; i < A.size(); ++i) A.host_data()[i] = 1.0f;
    for (int i = 0; i < B.size(); ++i) B.host_data()[i] = 2.0f;
    for (int i = 0; i < C.size(); ++i) C.host_data()[i] = 0.0f;

    // Copy data to device
    A.sync_device();
    B.sync_device();
    C.sync_device();

    // Initialize GEMM arguments
    CutlassGemm::Arguments args({M, N, K},  // Dimensions
                                {A.device_data(), A.stride(0)},  // A matrix
                                {B.device_data(), B.stride(0)},  // B matrix
                                {C.device_data(), C.stride(0)},  // C matrix
                                {C.device_data(), C.stride(0)},  // D matrix (output)
                                {1.0f, 0.0f});  // alpha and beta

    // Initialize GEMM
    CutlassGemm gemm_op;
    
    // Run GEMM
    cutlass::Status status = gemm_op(args);

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed with status " << static_cast<int>(status) << std::endl;
        return 1;
    }

    // Copy result back to host
    C.sync_host();

    // Print a small part of the result (modify as needed)
    std::cout << "Result (top-left 4x4 corner of C):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << C.at({i, j}) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}