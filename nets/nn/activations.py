from .module import Module


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return x * (x.data > 0) + x * self.alpha * (x.data <= 0)