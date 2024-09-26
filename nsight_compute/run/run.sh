# nvcc -arch=sm_86 ../example.cu -o example && ./example
# nvcc -arch=sm_86 -lineinfo ../example.cu -o example
nvcc -arch=sm_86 -g -G ../example.cu -o example