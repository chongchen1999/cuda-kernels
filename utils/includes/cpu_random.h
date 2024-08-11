#pragma once

#include <algorithm>
#include <ctime>
#include <cstdlib>

namespace randomTools {
    static unsigned int max_unsigned_int = 4294967295u;
    static unsigned short max_unsigned_short = 65535;

    template <typename T>
    void randomFill(T *data, int N, T min, T max, unsigned int seed = 233) {
        srand(seed);
        std::for_each(
            data, data + N, 
            [=](T &v) {
                float r = static_cast<float>(rand()) / RAND_MAX;
                v = min + static_cast<T>(r * (max - min));
                // printf("%.4f ", v);
            }
        );
        // puts("");
    }

    template <typename T>
    void randomNumber(T *data, T min, T max, unsigned int seed = 233) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        *data = min + static_cast<T>(r * (max - min));
    }

    unsigned int fastUnsignedInt() {
        static unsigned int x = 123456789u, y = 362436069u, z = 521288629u;
        unsigned int t;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        t = (y ^ z) + (y << 3) + (z >> 10);
        y = z;
        z = t;
        return x + t;
    }

    unsigned short fastUnsignedShort() {
        return fastUnsignedInt() % 65536;
    }

    template <typename T>
    void fastRandomFill(T *data, int N, T min, T max) {
        std::for_each(
            data, data + N, 
            [=](T &v) {
                float r = static_cast<float>(fastUnsignedInt()) / max_unsigned_int;
                v = min + static_cast<T>(r * (max - min));
                // printf("%.4f ", v);
            }
        );
        // puts("");
    }

}