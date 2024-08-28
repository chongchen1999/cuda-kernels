import triton
import triton.language as tl
import torch

# Triton matrix multiplication kernel
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, 
    stride_a_m, stride_a_k, 
    stride_b_k, stride_b_n, 
    stride_c_m, stride_c_n, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + (offs_am[:, None] * stride_a_m + (k + offs_k[None, :]) * stride_a_k))
        b = tl.load(B + (k + offs_k[:, None]) * stride_b_k + offs_bn[None, :] * stride_b_n)
        acc += tl.dot(a, b)
    
    tl.store(C + (offs_am[:, None] * stride_c_m + offs_bn[None, :] * stride_c_n), acc)

# Define matrix dimensions and block sizes
M, N, K = 128, 128, 128
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32

# Create input and output matrices
A = torch.randn((M, K), device='cuda', dtype=torch.float32)
B = torch.randn((K, N), device='cuda', dtype=torch.float32)
C = torch.zeros((M, N), device='cuda', dtype=torch.float32)

# Get matrix strides
stride_a_m, stride_a_k = A.stride()
stride_b_k, stride_b_n = B.stride()
stride_c_m, stride_c_n = C.stride()

# Launch Triton kernel
grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
kernel_fn = matmul_kernel[grid](
    A, B, C, M, N, K, 
    stride_a_m, stride_a_k, 
    stride_b_k, stride_b_n, 
    stride_c_m, stride_c_n, 
    BLOCK_SIZE_M=BLOCK_SIZE_M, 
    BLOCK_SIZE_N=BLOCK_SIZE_N, 
    BLOCK_SIZE_K=BLOCK_SIZE_K
)

# Get PTX code
ptx_code = kernel_fn.asm['ptx']

# Save PTX code to a text file
with open('matmul_kernel.ptx', 'w') as file:
    file.write(ptx_code)
