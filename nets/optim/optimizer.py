import numpy as np
class Optimizer:

    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)