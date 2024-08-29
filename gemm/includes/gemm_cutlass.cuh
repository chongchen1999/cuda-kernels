#pragma once



void sgemm_cutlass(
    float *host_A, float *host_B, float *hots_C, 
    int M, int K, int N,
    const int iterations = 1000
) {
}