import numpy as np

class Module:

    def parameters(self):
        params = []

        for attr in self.__dict__.values():

            if isinstance(attr, Module):
                params.extend(attr.parameters())

            elif hasattr(attr, "requires_grad"):
                params.append(attr)

        return params

    def zero_grad(self):

        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError