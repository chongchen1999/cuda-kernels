#cublas
nvcc -arch=sm_86 cublas_main.cu -lcublas -o cublas_main && ./cublas_main

# baseline
nvcc -arch=sm_86 -Xcompiler -fopenmp naive_wmma.cu main.cu -o naiva_wmma && ./naiva_wmma