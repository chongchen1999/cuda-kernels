TARGET := test_gemm

# Define the CUDA Toolkit path
# set your own CUDA_PATH if it's not in the default path
CUDA_PATH ?= /usr/local/cuda-12.5
CPATH = -I$(CUDA_PATH)/targets/x86_64-linux/include

# Compiler and linker flags
NVCCFLAGS = -arch=sm_86
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

all: $(TARGET)

# Rule to compile .cu files to .o
gemm.o: gemm.cu
	$(CUDA_PATH)/bin/nvcc $(NVCCFLAGS) -c $< -o $@

# Rule to compile .cpp files to .o
main.o: main.cpp
	g++ -c $< -o $@ $(CPATH)

# Rule to link object files into the executable
$(TARGET): gemm.o main.o
	g++ $^ -o $@ $(LDFLAGS)

# Phony target for cleaning
.PHONY: clean

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean 
