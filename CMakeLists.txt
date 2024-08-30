cmake_minimum_required(VERSION 3.10)
project(CutlassGEMM CUDA CXX)

find_package(CUTLASS REQUIRED)

add_executable(gemm gemm.cu)
target_link_libraries(gemm PRIVATE CUTLASS::cutlass)