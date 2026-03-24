import numpy as np

from nets.tensor.tensor import Tensor
from nets.nn import Linear, Sequential
from nets.optim import SGD
from nets.losses.cross_entropy import cross_entropy


# Create simple pattern
x_data = np.random.randn(200, 2)

# Rule-based labels
y_data = (x_data[:, 0] + x_data[:, 1] > 0).astype(int)

x = Tensor(x_data)
y = Tensor(y_data)


model = Sequential(
    Linear(2, 8),
    Linear(8, 2)
)

optimizer = SGD(model.parameters(), lr=0.1)


for epoch in range(50):

    logits = model(x)

    loss = cross_entropy(logits, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data}")