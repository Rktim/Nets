import numpy as np
from .module import Module
from .parameter import Parameter


class Linear(Module):

    def __init__(self, in_features, out_features):

        # Stable initialization
        self.weight = Parameter(
            np.random.randn(in_features, out_features) * np.sqrt(1 / in_features)
        )

        self.bias = Parameter(
            np.zeros((1, out_features))
        )

    def forward(self, x):
        return (x @ self.weight) + self.bias
    
    def __repr__(self):
        return f"Linear({self.weight.data.shape[0]}, {self.weight.data.shape[1]})"