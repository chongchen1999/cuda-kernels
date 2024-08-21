#include <iostream>
#include "../utils/includes/cpu_random.h"

int main() {
    const int N = 1 << 25;
    int *data = new int[N];

    randomTools::fastRandomFill(data, N, 0, 1 << 30);
    
    return 0;
}