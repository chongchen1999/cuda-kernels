# CUDA Operators Implementation

This project implements several common deep learning operators using CUDA. The focus is on optimizing performance to achieve results comparable to established libraries like cuBLAS, cuDNN, and CUB.

## Implemented Operators

- **General Matrix Multiply (GEMM)**
- **Reduction**
- **Element-wise Operations**
- **Fused Operations**

## Optimization Techniques

### 1. GEMM (General Matrix Multiply)
- **Matrix Tiling**: The matrix is divided into smaller tiles, with each block and thread responsible for computing a tile.
- **Shared Memory**: Source matrix data is loaded into shared memory in a block-wise manner.
- **Thread-level Computation**: Each thread computes the outer product of rows/columns loaded into registers from shared memory.
- **Double Buffering**: Data prefetching is used to hide memory latency, and all memory access operations are vectorized.
- **Performance**: Achieves an average of 90% of cuBLAS performance for non-special matrix shapes.

### 2. Reduction
- **Warp-level Optimization**: Uses warp shuffle instructions to exploit the SIMT characteristics, reducing thread synchronization overhead.
- **Block-level Optimization**: Shared memory is used to store the reduction results of each warp, followed by a final warp-level reduction to compute the block-level result.
- **Performance**: Achieves performance on par with CUB.

### 3. Element-wise Operations
- **Vectorized Memory Access**: Memory access time is reduced by using vectorized operations.
- **Performance**: Achieves performance comparable to cuBLAS for operations like vector addition.

### 4. Fused Operations
- **Memory Access Reduction**: Minimizes memory access by avoiding storage of intermediate results, optimizing computation order, and improving data locality.
- **Softmax Example**:
  - For smaller category sizes, each warp processes one row of the batch.
  - For larger category sizes, each block processes one row of the batch.
- **Performance**: Reaches cuDNN-level performance.

## Learnings and Key Concepts
- **Parallel Algorithm Design**: Developed a strong understanding of parallel algorithm design principles.
- **GPU Architecture**: Gained in-depth knowledge of GPU architecture.
- **Key CUDA Programming Concepts**:
  - **Aligned Access**
  - **Coalesced Access**
  - **Bank Conflict**
  - **Warp Divergence**

## Conclusion
This project serves as a comprehensive exercise in CUDA programming and performance optimization. The implementations achieve high performance, comparable to industry-standard libraries, and provide a solid foundation for further exploration of GPU-accelerated computing.
