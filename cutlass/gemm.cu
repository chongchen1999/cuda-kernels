#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm.h>

int main() {
    // 定义矩阵维度
    int N = 512;
    int M = 512;
    int K = 512;

    // 定义矩阵数据类型
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    // 定义GEMM核
    using Gemm = cutlass::gemm::device::Gemm<
        ElementOutput,   // Data type of A elements
        cutlass::layout::RowMajor,  // Layout of A matrix
        ElementOutput,   // Data type of B elements
        cutlass::layout::RowMajor,  // Layout of B matrix
        ElementOutput,   // Data type of C elements
        cutlass::layout::RowMajor,  // Layout of C matrix
        ElementAccumulator,  // Data type of accumulator
        cutlass::arch::OpClassSimt,  // Operation class (SIMT, TensorOp, etc.)
        cutlass::arch::Sm50   // Architecture (e.g., Sm50, Sm70, Sm75, etc.)
    >;

    // 创建GEMM问题大小
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // 分配主机内存
    ElementOutput *A = new ElementOutput[N * K];
    ElementOutput *B = new ElementOutput[K * M];
    ElementOutput *C = new ElementOutput[N * M];

    // 初始化输入矩阵
    for (int i = 0; i < N * K; i++) {
        A[i] = static_cast<ElementOutput>(1);
    }
    for (int i = 0; i < K * M; i++) {
        B[i] = static_cast<ElementOutput>(1);
    }
    for (int i = 0; i < N * M; i++) {
        C[i] = static_cast<ElementOutput>(0);
    }

    // 创建GEMM操作
    typename Gemm::Arguments arguments{
        problem_size,
        {A, K},
        {B, M},
        {C, M},
        {C, M},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}  // alpha, beta
    };

    // 创建GEMM对象
    Gemm gemm_op;

    // 调用GEMM运算
    cutlass::Status status = gemm_op(arguments);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM computation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // 打印输出矩阵中的前10个值
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // 释放内存
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
