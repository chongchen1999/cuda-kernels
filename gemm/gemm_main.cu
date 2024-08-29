#include "includes/gemm_thread.cuh"
#include "includes/gemm_cublas.cuh"
#include <cmath>
#include <numeric>

void init_matrix(float *A, int M, int N, float value = 1.0f) {
    for (int i = 0; i < M * N; ++i) {
        A[i] = value * (float)rand() / RAND_MAX;
    }

    float sum = std::accumulate(A, A + M * N, 0.0f);
    float mean = sum / (M * N);

    float sq_sum = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        sq_sum += (A[i] - mean) * (A[i] - mean);
    }
    float stddev = std::sqrt(sq_sum / (M * N));

    for (int i = 0; i < M * N; ++i) {
        A[i] = (A[i] - mean) / stddev;
    }
}

bool check_reult(float *A, float *B, int size) {
    for (int i = 0; i < size; ++i) {
        const float eps = 1e-2;
        const float diff = fabs(A[i] - B[i]);
        const float val = fmax(fabs(A[i]), fabs(B[i])) + eps;
        if (diff > eps * val && diff > eps) {
            printf("%d: %f != %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    float *host_A, *host_B;

    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    host_A = (float *)malloc(M * K * sizeof(float));
    host_B = (float *)malloc(K * N * sizeof(float));

    init_matrix(host_A, M, K, 0.1);
    init_matrix(host_B, K, N, 0.1);

    float *host_C = (float *)malloc(M * N * sizeof(float));
    gemm_thread::sgemm_kernel(host_A, host_B, host_C, M, K, N);

    float *host_C_cublas = (float *)malloc(M * N * sizeof(float));
    gemm_cublas::sgemm_cublas(host_A, host_B, host_C_cublas, M, K, N);

    if (check_reult(host_C, host_C_cublas, M * N)) {
        printf("result is right\n");
    } else {
        printf("result is wrong\n");
    }

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_C_cublas);
    return 0;
}