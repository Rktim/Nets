import numpy as np

from nets.tensor.tensor import Tensor
from nets.nn import Linear, ReLU, Sequential
from nets.optim import SGD
from nets.losses import mse
from nets.trainer.trainer import Trainer


# dataset: y = 2x + 1
x_data = np.random.randn(100, 1)
y_data = x_data ** 2  # nonlinear

x = Tensor(x_data)
y = Tensor(y_data)


# model
model = Sequential(
    Linear(1, 8),
    ReLU(),
    Linear(8, 1)
)

# optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# trainer
trainer = Trainer(model, optimizer, mse)

trainer.fit(x, y, epochs=100)