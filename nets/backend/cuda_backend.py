import cupy as cp
from .backend_base import Backend


class CUDABackend(Backend):

    name = "cuda"

    def array(self, data):
        return cp.array(data, dtype=cp.float32)

    def zeros(self, shape):
        return cp.zeros(shape, dtype=cp.float32)

    def ones(self, shape):
        return cp.ones(shape, dtype=cp.float32)

    def matmul(self, a, b):
        return cp.matmul(a, b)

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        return a / b

    def sum(self, a, axis=None):
        return cp.sum(a, axis=axis)

    def mean(self, a, axis=None):
        return cp.mean(a, axis=axis)

    def exp(self, a):
        return cp.exp(a)

    def log(self, a):
        return cp.log(a)

    def reshape(self, a, shape):
        return cp.reshape(a, shape)

    def transpose(self, a):
        return cp.transpose(a)