#pragma once
#include <cmath> // For std::abs and std::fabs
#include <limits> // For std::numeric_limits

template <typename T>
struct Equal {
    bool operator()(T a, T b) {
        return a == b;
    }
};

template <>
struct Equal<float> {
    bool operator()(float a, float b, float eps = 1e-3) {
        return std::fabs(a - b) <= eps * std::fmax(std::fabs(a), std::fabs(b));
    }
};


template <typename T>
bool checkResult(T *a, T *b, int N) {
    for (int i = 0; i < N; ++i) {
        if (!Equal<T>()(a[i], b[i])) {
            printf("a[%d] = %f, b[%d] = %f\n", i, a[i], i, b[i]);
            return false;
        }
    }
    return true;
}