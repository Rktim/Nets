import numpy as np


class SGD:

    def __init__(self, params, lr=0.01, momentum=0.0):

        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.vel = [np.zeros_like(p.data) for p in params]

    def step(self):

        for i, p in enumerate(self.params):

            if p.grad is None:
                continue

            self.vel[i] = self.momentum * self.vel[i] - self.lr * p.grad
            p.data += self.vel[i]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)