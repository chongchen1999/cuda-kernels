import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

print('loading')

# Build shared library for loading the CUDA kernel as a python module
flash_attention = load(name='flash_attention', sources=['main.cpp', 'flash_attention.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
head_num = 12
seq_len = 64
head_size = 64

print('loaded!')

q = torch.randn(batch_size, head_num, seq_len, head_size).cuda()
k = torch.randn(batch_size, head_num, seq_len, head_size).cuda()
v = torch.randn(batch_size, head_num, seq_len, head_size).cuda()

def manual_attention(q, k, v):
    att = torch.matmul(q, k.transpose(-2, -1)).mul_(1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    att = torch.matmul(att, v)
    return att

print('=== profiling manual attention ===')

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    manual_result = manual_attention(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


print('=== profiling flash attention === ')

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    flash_attention_result = flash_attention.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print(manual_result)
# print(flash_attention_result)

print('attn values sanity check:', torch.allclose(flash_attention_result, manual_result, rtol=0, atol=1e-02))