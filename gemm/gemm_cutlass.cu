#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm.h>
#include <vector>
#include <iostream>

int main() {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator
    >;

    int M = 128;  // Number of rows in A and C
    int N = 128;  // Number of columns in B and C
    int K = 128;  // Number of columns in A and rows in B

    std::vector<float> matrixA(M * K, 1.0f);
    std::vector<float> matrixB(K * N, 1.0f);
    std::vector<float> matrixC(M * N, 0.0f);

    cutlass::TensorRef<ElementInputA, LayoutInputA> tensorA(matrixA.data(), M);
    cutlass::TensorRef<ElementInputB, LayoutInputB> tensorB(matrixB.data(), N);
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensorC(matrixC.data(), N);

    typename Gemm::Arguments args(
        {M, N, K},
        {tensorA.data(), M},
        {tensorB.data(), K},
        {tensorC.data(), N},
        {tensorC.data(), N},
        {1.0f, 0.0f}
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed." << std::endl;
        return -1;
    }

    // Check results or use matrixC as needed
    for (int i = 0; i < M * N; ++i) {
        std::cout << matrixC[i] << " ";
    }

    return 0;
}
