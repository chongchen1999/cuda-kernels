import torch
import triton
import triton.language as tl
import time

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# Initialize tensors
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y

# Measure performance for Torch
torch_times = []
for _ in range(1000):
    start_time = time.time()
    output_torch = x + y
    torch.cuda.synchronize()  # Ensure that all operations are completed
    torch_times.append(time.time() - start_time)

# Measure performance for Triton
triton_times = []
for _ in range(1000):
    start_time = time.time()
    output_triton = add(x, y)
    torch.cuda.synchronize()  # Ensure that all operations are completed
    triton_times.append(time.time() - start_time)

# Print results
print(f'Torch average time: {sum(torch_times) * 1000 / len(torch_times):.6f} ms')
print(f'Triton average time: {sum(triton_times) * 1000 / len(triton_times):.6f} ms')

# Optional: Verify correctness
max_difference = torch.max(torch.abs(output_torch - output_triton))
print(f'The maximum difference between torch and triton is {max_difference:.6f}')
