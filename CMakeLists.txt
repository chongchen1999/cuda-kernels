cmake_minimum_required(VERSION 3.10)
project(CutlassGEMM CUDA CXX)

add_executable(gemm gemm.cu)
target_link_libraries(gemm PRIVATE CUTLASS::cutlass)