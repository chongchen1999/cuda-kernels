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

int M = 5376;
int N = 5376;
int K = 2048;

int iterations = 200;

#define MAX(a, b) (a) > (b) ? (a) : (b)

const int block_tile_m = 128;
const int block_tile_n = 128;
const int block_tile_k = 32;