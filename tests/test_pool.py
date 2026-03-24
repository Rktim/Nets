import numpy as np
from nets.tensor.tensor import Tensor
from nets.nn.maxpool2d import MaxPool2D


x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)

pool = MaxPool2D(2)

out = pool(x)

print("Output:", out.data.shape)

loss = out.sum()
loss.backward()

print("Grad OK:", x.grad.shape)