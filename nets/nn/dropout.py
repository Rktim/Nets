import numpy as np
from .module import Module


class Dropout(Module):

    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):

        if not self.training:
            return x

        mask = (np.random.rand(*x.data.shape) > self.p) / (1 - self.p)
        return x * mask