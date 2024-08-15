#include <stdio.h>
#include <cuda.h>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void memoryBandwidth (float* A,  float* B, float* C, int N) {
	const int gid = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    const int grid_stride = (blockDim.x * gridDim.x) << 2;

    #pragma unroll
	for(int i = gid; i < N; i += grid_stride) {
		float4 a1 = *reinterpret_cast<float4 *>(A + i);
		// float4 b1 = *reinterpret_cast<float4 *>(B + i);
		/*float4 c1 = {
            a1.x + b1.x,
            a1.y + b1.y,
            a1.z + b1.z,
            a1.w + b1.w
        };*/
		*reinterpret_cast<float4 *>(C + i) = a1;
	}
}

void vec_add_cpu(float *x, float *y, float *z, int N) {
    for (int i = 0; i < 20; i++) z[i] = y[i] + x[i];
}

int main() {
    srand(233666);
    constexpr int length = 1 << 26;
    constexpr int length_per_iter = 1 << 22; // exceed L2 cache size
    constexpr int iterations = length / length_per_iter;
    constexpr int thread_num = 256;
	float *A = (float *)malloc(length * sizeof(float));
	float *B = (float *)malloc(length * sizeof(float));
	float *C = (float *)malloc(length * sizeof(float));

	float *device_A, *device_B, *device_C;
	float milliseconds = 0;

	for (int i = 0; i < length; ++i) {
		A[i] = (float)i;
		B[i] = (float)i;
	}

	cudaMalloc((void **)&device_A, length * sizeof(float));
	cudaMalloc((void **)&device_B, length * sizeof(float));
	cudaMalloc((void **)&device_C, length * sizeof(float));

	cudaMemcpy(device_A, A, length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_B, B, length * sizeof(float), cudaMemcpyHostToDevice);
  
	const int vec_block_nums = length_per_iter / (thread_num << 2);
    int offsets[iterations];
    for (int i = 0; i < iterations; ++i) {
        offsets[i] = i;
    }
    for (int i = 1; i < iterations; ++i) {
        const int pre_idx = std::rand() % i;
        std::swap(offsets[i], offsets[pre_idx]);
    }

    //warm up to occupy L2 cache
	printf("warm up start\n");
	memoryBandwidth<<<vec_block_nums, thread_num>>>(device_A, device_B, device_C, length_per_iter);
	printf("warm up end\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (int i = 0; i < iterations; ++i) {
        const int offset = offsets[i] * length_per_iter;
		memoryBandwidth<<<vec_block_nums, thread_num>>>(
            device_A + offset,
            device_B + offset,
            device_C + offset,
            length_per_iter
        );
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(C, device_C, length*sizeof(float), cudaMemcpyDeviceToHost);
	float* C_cpu_res = (float *) malloc(20 * sizeof(float));
	vec_add_cpu(A, B, C_cpu_res, length);

	/* check GPU result with CPU*/
	for (int i = 0; i < 20; ++i) {
		if (fabs(C_cpu_res[i] - C[i]) > 1e-3) {
			printf("Result verification failed at element index %d!\n", i);
            // return 0;
		}
	}
	unsigned int bytes = length * 4;
    const float giga_bytes = static_cast<float>(bytes) / 1024.f / 1024.f / 1024.f;
    const float seconds = milliseconds / 1000.f;
	printf("Memory BandWidth = %f (GB/sec)\n", 2.0f * giga_bytes / seconds);
  	cudaFree(device_A);
  	cudaFree(device_B);
  	cudaFree(device_C);

  	free(A);
  	free(B);
  	free(C);
  	free(C_cpu_res);
    return 0;
}
