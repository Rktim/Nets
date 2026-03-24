import numpy as np
from nets.tensor.tensor import Tensor
from nets.nn.conv2d import Conv2D


x = Tensor(np.random.randn(4, 1, 28, 28), requires_grad=True)

conv = Conv2D(1, 8, 3)

out = conv(x)

print("Output:", out.data.shape)

loss = out.sum()
loss.backward()

print("Grad OK:", x.grad.shape)