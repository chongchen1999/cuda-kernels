#post train dynamic quantization

import torch

class simple_nn(torch.nn.Module):
    def __init__(self):
        super(simple_nn, self).__init__()
        self.linear1 = torch.nn.Linear(3, 3, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3, 1, bias=False)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        return outputs

# build data
weights = torch.tensor([[1.1], [2.2], [3.3]])
torch.manual_seed(123)
train_features = torch.randn(12000, 3)
train_labels = torch.matmul(train_features, weights)

test_features = torch.randn(1000, 3)
test_labels = torch.matmul(test_features, weights)

model = simple_nn()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

model.train()  # Set the model to training mode

for i in range(100):
    preds = model(train_features)  # Use train_features during training
    loss = torch.nn.functional.mse_loss(preds, train_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    preds = model(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"float32 model testing loss: {mse.item():.6f}")

# Apply dynamic quantization
model_int8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print(model_int8)

with torch.no_grad():
    preds = model_int8(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"int8 model testing loss: {mse.item():.6f}")

print("float32 model linear1 parameter:\n", model.linear1.weight)
print("int8 model linear1 parameter(int8):\n", torch.int_repr(model_int8.linear1.weight()))
print("int8 model linear1 parameter:\n", model_int8.linear1.weight())
print(model_int8.linear1)
