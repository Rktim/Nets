import numpy as np
from nets.nn.module import Module
from nets.tensor.tensor import Tensor


class RNN(Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # weights
        self.Wxh = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.Whh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.bh = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    def forward(self, x):

        batch, seq_len, _ = x.data.shape

        h = Tensor(np.zeros((batch, self.hidden_size)), requires_grad=False)

        for t in range(seq_len):
            xt = x[:, t, :]
            h = (xt @ self.Wxh + h @ self.Whh + self.bh).tanh()

        return h