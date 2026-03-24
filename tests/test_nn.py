from nets.tensor.tensor import Tensor
from nets.nn import Linear, ReLU, Sequential


# simple model
model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
)

# input
x = Tensor([[1.0, 2.0]], requires_grad=True)

# forward
y = model(x)

print("Output:", y.data)

# backward
loss = y.mean()
loss.backward()

# check gradients
for p in model.parameters():
    print("Grad shape:", p.grad.shape)