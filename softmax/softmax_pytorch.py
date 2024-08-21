import torch
import torch.nn.functional as F

# Create a random tensor on the GPU
X = torch.randn(3000, 1024, device='cuda')

# Create CUDA events to measure time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Start timing
start_event.record()

# Loop softmax 1000 times
for _ in range(1000):
    Y = F.softmax(X, dim=-1)

# End timing
end_event.record()

# Wait for the events to be recorded
torch.cuda.synchronize()

# Calculate elapsed time in milliseconds
elapsed_time_ms = start_event.elapsed_time(end_event)

print(f"Time taken for 1000 softmax operations: {elapsed_time_ms / 1000.0:.4f} ms")
