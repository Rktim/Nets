from .module import Module


class Flatten(Module):

    def forward(self, x):
        batch_size = x.data.shape[0]
        return x.reshape((batch_size, -1))