#pragma once

#include <algorithm>

namespace cpu_softmax {

    template <typename T>
    void launchSoftmax(T *output, T *input, int M, int N) {
        for (int i = 0; i < M; ++i) {
            T max = input[i * N];
            for (int j = 1; j < N; ++j) {
                max = std::max(input[i * N + j], max);
            }
            T sum = T(0);
            for (int j = 0; j < N; ++j) {
                T &temp = output[i * N + j];
                temp = std::exp(input[i * N + j] - max);
                sum += temp;
            }
            for (int j = 0; j < N; ++j) {
                output[i * N + j] /= sum;
            }
        }
    }
}