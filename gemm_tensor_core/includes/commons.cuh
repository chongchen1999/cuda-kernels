#pragma once

#define CUDA_CHECK(status)                                                \
{                                                                         \
    cudaError_t error = status;                                           \
    if (error != cudaSuccess)                                             \
    {                                                                     \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                    << " at line: " << __LINE__ << std::endl;             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

inline int M = 5376;
inline int N = 5376;
inline int K = 2048;
inline constexpr int warpSize = 32;

inline int iterations = 200;

#define MAX(a, b) (a) > (b) ? (a) : (b)

const int block_tile_m = 128;
const int block_tile_n = 128;
const int block_tile_k = 32;