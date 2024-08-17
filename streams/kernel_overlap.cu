#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void floatAddOne(float *buffer, int n) {
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = gid; i < n; i += stride) {
        buffer[i] += 1.0f;
    }
}

void launchFloatAddOne(
    float *buffer,
    int n,
    dim3 &threads_per_block,
    dim3 &blocks_per_grid,
    cudaStream_t stream
) {
    floatAddOne<<<blocks_per_grid, threads_per_block, 0, stream>>>(buffer, n);
}

int main(int argc, char **argv) {
    const size_t buffer_size = 1024 * 10240;
    const size_t num_streams = 5;

    dim3 threads_per_block(512);

    // Try different values for blocks_per_grid and see stream result on nsight.
    // 1, 2, 4, 8, 16, 32, 1024
    dim3 blocks_per_grid(32);

    std::vector<float *> d_buffers(num_streams);
    std::vector<cudaStream_t> streams(num_streams);

    for (auto &d_buffer : d_buffers) {
        cudaMalloc(reinterpret_cast<void **>(&d_buffer), buffer_size * sizeof(float));
        cudaMemset(d_buffer, 0, buffer_size * sizeof(float));
    }

    // Create num_streams streams
    for (auto &stream : streams) {
        cudaStreamCreate(&stream);
    }

    // Each independent kernel running on each stream to parallelize
    for (size_t i = 0; i < num_streams; ++i) {
        launchFloatAddOne(
            d_buffers[i],
            buffer_size,
            threads_per_block,
            blocks_per_grid,
            streams[i]
        );
    }

    // Host wait for each stream to complete
    for (auto &stream : streams) {
        cudaStreamSynchronize(stream);
    }

    for (auto &d_buffer : d_buffers) {
        cudaFree(d_buffer);
    }

    // Destroy CUDA streams
    for (auto &stream : streams) {
        cudaStreamDestroy(stream);
    }

    return 0;
}
