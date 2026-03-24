import numpy as np
_grad_enabled = True

class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc, tb):
        global _grad_enabled
        _grad_enabled = self.prev

def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:

    def __init__(self, data, requires_grad=False):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data.astype(np.float32)
        self.requires_grad = requires_grad and _grad_enabled

        self.grad = np.zeros_like(self.data) if self.requires_grad else None

        self._prev = set()
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # -------------------------
    # BASIC OPS
    # -------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad -= unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data / other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast((1 / other.data) * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad -= unbroadcast((self.data / (other.data ** 2)) * out.grad,
                                         other.data.shape)

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __neg__(self):
        return self * -1

    def __matmul__(self, other):

        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    # -------------------------
    # REDUCTIONS
    # -------------------------

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data.size) * np.ones_like(self.data) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    # -------------------------
    # ACTIVATIONS
    # -------------------------

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (sig * (1 - sig)) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    # -------------------------
    # EXP / LOG
    # -------------------------

    def exp(self):
        exp_data = np.exp(self.data)
        out = Tensor(exp_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += exp_data * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-9), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1 / (self.data + 1e-9)) * out.grad

        out._backward = _backward
        out._prev = {self}
        return out

    # -------------------------
    # SHAPE OPS (IMPORTANT FOR TRANSFORMERS)
    # -------------------------

    def transpose(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = _backward
        out._prev = {self}
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        out._prev = {self}
        return out

    # -------------------------
    # BACKPROP
    # -------------------------

    def backward(self):

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # reset gradients
        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
            
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad += grad

        out._backward = _backward
        out._prev = {self}

        return out