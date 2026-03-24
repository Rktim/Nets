import numpy as np
from .backend_base import Backend


class CPUBackend(Backend):

    name = "cpu"

    def array(self, data):
        return np.array(data, dtype=np.float32)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float32)

    def ones(self, shape):
        return np.ones(shape, dtype=np.float32)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a / b

    def sum(self, a, axis=None):
        return np.sum(a, axis=axis)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)

    def reshape(self, a, shape):
        return np.reshape(a, shape)

    def transpose(self, a):
        return np.transpose(a)