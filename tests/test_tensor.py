from nets.tensor.tensor import Tensor


# simple forward + backward

x = Tensor([[1.0, 2.0]], requires_grad=True)
w = Tensor([[3.0], [4.0]], requires_grad=True)

y = x @ w
loss = y.mean()

loss.backward()

print("Output:", y.data)
print("Grad w:", w.grad)
print("Grad x:", x.grad)