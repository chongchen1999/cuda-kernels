#cublas
nvcc -arch=sm_86 ../cublas_main.cu -lcublas -o cublas_main && ./cublas_main

# nvcc -arch=sm_86 -Xcompiler -fopenmp naive_wmma.cu main.cu -o naiva_wmma && ./naiva_wmma
# nvcc -arch=sm_86 -Xcompiler -fopenmp ../wmma_4stage.cu ../main.cu -o wmma_4stage && ./wmma_4stage
nvcc -arch=sm_86 -Xcompiler -fopenmp ../wmma_4stage_asynccp.cu ../main.cu -o wmma_4stage_asynccp && ./wmma_4stage_asynccp stages 4
# nvcc -arch=sm_86 -Xcompiler -fopenmp ../mma_4stage.cu ../main.cu -o mma_4stage && ./mma_4stage stages 4
# nvcc -arch=sm_86 -Xcompiler -fopenmp ../mma_4stage_swizzle.cu ../main.cu -o mma_4stage_swizzle && ./mma_4stage_swizzle stages 4 multi_threading 2