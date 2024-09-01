import torch

class simple_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear1 = torch.nn.Linear(3, 3, bias=False)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3, 1, bias=False)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, inputs):
        q_inputs = self.quant(inputs)
        outputs = self.linear1(q_inputs)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        f_outputs = self.dequant(outputs)
        return f_outputs

# build data
weights = torch.tensor([[1.1], [2.2], [3.3]])
torch.manual_seed(123)
train_features = torch.randn(12000, 3)
train_labels = torch.matmul(train_features, weights)

test_features = torch.randn(1000, 3)
test_labels = torch.matmul(test_features, weights)

model = simple_nn().train()

model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_prepared = torch.ao.quantization.prepare_qat(model)

optimizer = torch.optim.Adam(model_prepared.parameters(), lr = 0.1)
for i in range(100):
    preds = model_prepared(train_features)
    loss = torch.nn.functional.mse_loss(preds, train_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model_prepared.eval()
with torch.no_grad():
    preds = model_prepared(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"float32 model testing loss:{mse.item():.6f}")

model_int8 = torch.ao.quantization.convert(model_prepared).eval()
with torch.no_grad():
    preds = model_int8(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"int8 model testing loss:{mse.item():.6f}")

print("float32 model linear1 parameter:\n", model.linear1.weight)
print("int8 model linear1 parameter(int8):\n", torch.int_repr(model_int8.linear1.weight()))
print("int8 model linear1 parameter:\n", model_int8.linear1.weight())