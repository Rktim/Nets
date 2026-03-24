import numpy as np


class RMSprop:

    def __init__(self, params, lr=0.001, beta=0.9):

        self.params = params
        self.lr = lr
        self.beta = beta

        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):

        for i, p in enumerate(self.params):

            if p.grad is None:
                continue

            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (p.grad ** 2)

            p.data -= self.lr * p.grad / (np.sqrt(self.v[i]) + 1e-8)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)