import numpy as np


class Adam:

    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999):

        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

        self.t = 0

    def step(self):

        self.t += 1

        for i, p in enumerate(self.params):

            if p.grad is None:
                continue

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)