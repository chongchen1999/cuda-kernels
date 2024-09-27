nvcc -arch=sm_86 -o reduce_sum_atomic_v2 ../reduce_sum_atomic_v2.cu && ./reduce_sum_atomic_v2
nvcc -arch=sm_86 -o reduce_sum_v7 ../reduce_sum_v7.cu && ./reduce_sum_v7
nvcc -arch=sm_86 -o reduce_sum_atomic_v1 ../reduce_sum_atomic_v1.cu && ./reduce_sum_atomic_v1
# nvcc -arch=sm_86 -o reduce_sum_atomic_v3 ../reduce_sum_atomic_v3.cu && ./reduce_sum_atomic_v3