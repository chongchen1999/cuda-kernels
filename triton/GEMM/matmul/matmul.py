import torch
import triton
from .kernel import matmul_kernel
from .activation import leaky_relu

def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "维度符合矩阵相乘要求"
    assert a.is_contiguous(), "矩阵A必须是连续的"
    assert b.is_contiguous(), "矩阵B必须是连续的"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # print("ready for launching matmul_kernel:")
    # print(a.stride(0))
    # print(a.stride(1))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c
