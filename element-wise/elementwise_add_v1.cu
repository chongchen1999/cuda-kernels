#include <iostream>
#include <cuda_runtime.h>

const int N = 1 << 5;
const int iterations = 1000;

__global__ void add_v1(float *a, float *b, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        result[tid] = a[tid] + b[tid];
    }
}

bool check_result(float *a, float *b) {
    for (int i = 0; i < N; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void check_cuda_error(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *host_a = (float *) malloc(N * sizeof(float));
    float *host_b = (float *) malloc(N * sizeof(float));
    float *host_result = (float *) malloc(N * sizeof(float));
    float *cpu_result = (float *) malloc(N * sizeof(float));

    float *device_a;
    float *device_b;
    float *device_result;

    cudaError_t err;

    err = cudaMalloc((void **) &device_a, N * sizeof(float));
    check_cuda_error(err, "cudaMalloc device_a");
    err = cudaMalloc((void **) &device_b, N * sizeof(float));
    check_cuda_error(err, "cudaMalloc device_b");
    err = cudaMalloc((void **) &device_result, N * sizeof(float));
    check_cuda_error(err, "cudaMalloc device_result");

    for (int i = 0; i < N; ++i) {
        host_a[i] = i;
        host_b[i] = (N % (i + 1)) * 1.0f; // Cast to float
        cpu_result[i] = host_a[i] + host_b[i];
    }

    std::cout << "Initializing data\n";

    err = cudaMemcpy(device_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error(err, "cudaMemcpy host_a to device_a");
    err = cudaMemcpy(device_b, host_b, N * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error(err, "cudaMemcpy host_b to device_b");

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    dim3 Grid(grid_size);
    dim3 Block(block_size);

    for (int i = 0; i < iterations; ++i) {
        add_v1<<<Grid, Block>>>(device_a, device_b, device_result);
        err = cudaGetLastError();
        check_cuda_error(err, "Kernel launch");
    }

    err = cudaMemcpy(host_result, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error(err, "cudaMemcpy device_result to host_result");

    if (!check_result(host_result, cpu_result)) {
        printf("Error\n");
        for (int i = 0; i < N; ++i) {
            std::cout << host_result[i] << " ";
        }
        std::cout << "\n";

        for (int i = 0; i < N; ++i) {
            std::cout << cpu_result[i] << " ";
        }
        std::cout << "\n";
    } else {
        printf("Success\n");
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_result);
    free(host_a);
    free(host_b);
    free(host_result);
    free(cpu_result);

    return 0;
}
