import torch
import torch.nn.functional as F

def softmax_cudnn(X):
    return F.softmax(X, dim=-1)

# 测试
X = torch.randn(128, 1024, device='cuda')
Y = softmax_cudnn(X)
