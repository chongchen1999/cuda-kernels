#!/bin/sh

nvcc -arch=sm_86 -o elementwise_add_v1 ../elementwise_add_v1.cu && ./elementwise_add_v1
nvcc -arch=sm_86 -o elementwise_add_v2 ../elementwise_add_v2.cu && ./elementwise_add_v2
nvcc -arch=sm_86 -o elementwise_add_v3 ../elementwise_add_v3.cu && ./elementwise_add_v3

# nvcc -arch=sm_86 -o elementwise_add_v4 ../elementwise_add_v4.cu && ./elementwise_add_v4