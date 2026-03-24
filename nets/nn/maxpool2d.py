import numpy as np
from .module import Module
from nets.tensor.tensor import Tensor


class MaxPool2D(Module):

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def forward(self, x):

        N, C, H, W = x.data.shape
        K = self.kernel_size
        S = self.stride

        # Ensure dimensions divisible (simplification)
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        # ---- reshape trick (fast)
        x_reshaped = x.data.reshape(
            N,
            C,
            H_out,
            K,
            W_out,
            K
        )

        # swap axes to group patches
        x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5)

        # ---- max pooling
        out = x_reshaped.max(axis=(4, 5))

        out_tensor = Tensor(out, requires_grad=True)

        # ---- store mask for backward
        max_mask = (x_reshaped == out[..., None, None])

        def _backward():

            dx = np.zeros_like(x_reshaped)

            # distribute gradient
            grad = out_tensor.grad[..., None, None]
            dx += max_mask * grad

            # reshape back
            dx = dx.transpose(0,1,2,4,3,5).reshape(x.data.shape)

            if x.requires_grad:
                x.grad += dx

        out_tensor._backward = _backward
        out_tensor._prev = {x}

        return out_tensor