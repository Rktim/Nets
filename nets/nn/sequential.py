from .module import Module


class Sequential(Module):

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):

        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params