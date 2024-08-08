#include <cstdio>
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <chrono>

static const int iterations = 4;

void initMatrix(float *A, int size) {
    std::for_each(
        A, A + size,
        [](float &a) {
            a = static_cast<float>(std::rand()) / RAND_MAX;
        }
    );
}

void naiveMatmul(float *C, float *A, float *B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float val_a = A[i * K + k];
            int base_offset = k * N;
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += val_a * B[base_offset + j];
            }
        }
    }
}

void TiledMatmul(float *C, float *A, float *B, int M, int N, int K) {
    int tile_size = 16;
    for (int tile_row = 0; tile_row < M; tile_row += tile_size) {
        for (int tile_col = 0; tile_col < N; tile_col += tile_size) {
            for (int tile_k = 0; tile_k < K; tile_k += tile_size) {
                float *tile_a = A + tile_row * K + tile_k;
                float *tile_b = B + tile_k * N + tile_col;
                float *tile_c = C + tile_row * N + tile_col;
                for (int i = 0; i < tile_size; ++i) {
                    for (int k = 0; k < tile_size; ++k) {
                        float val_a = tile_a[i * K + k];
                        int base_offset = k * N;
                        for (int j = 0; j < tile_size; ++j) {
                            tile_c[i * N + j] += val_a * tile_b[base_offset + j];
                        }
                    }
                }
            }
        }
    }
}

void testNaiveMul(float *C, float *A, float *B, int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        naiveMatmul(C, A, B, M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Naive: " << duration / iterations << " ms!" << std::endl;
}

void testTiledMul(float *C, float *A, float *B, int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        TiledMatmul(C, A, B, M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Tiled: " << duration / iterations << " ms!" << std::endl;
}

int main(int argc, char *argv[]) {
    std::srand(233u);
    int M = 512;
    int K = 512;
    int N = 512;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    initMatrix(A, M * K);
    initMatrix(B, K * N);

    testNaiveMul(C, A, B, M, N, K);
    testTiledMul(C, A, B, M, N, K);

    return 0;
}