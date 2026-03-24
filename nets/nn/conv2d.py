import numpy as np
from .module import Module
from .parameter import Parameter
from nets.tensor.tensor import Tensor


def im2col(x, K, stride, padding):
    N, C, H, W = x.shape

    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))

    H_p, W_p = x.shape[2], x.shape[3]

    out_h = (H_p - K) // stride + 1
    out_w = (W_p - K) // stride + 1

    # ---- stride trick (no loops)
    shape = (N, C, K, K, out_h, out_w)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2],
        x.strides[3],
        x.strides[2]*stride,
        x.strides[3]*stride
    )

    patches = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides
    )

    cols = patches.reshape(N, C*K*K, out_h*out_w)
    cols = cols.transpose(0, 2, 1)  # (N, L, C*K*K)

    return cols, x.shape

def col2im(cols, input_shape, kernel_size):
    N, C, H, W = input_shape
    K = kernel_size

    out_h = H - K + 1
    out_w = W - K + 1

    output = np.zeros(input_shape, dtype=np.float32)

    col_idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = cols[:, col_idx, :].reshape(N, C, K, K)
            output[:, :, i:i+K, j:j+K] += patch
            col_idx += 1

    return output


class Conv2D(Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        )
        self.bias = Parameter(np.zeros((out_channels,)))

    def forward(self, x):

        N, C, H, W = x.data.shape
        K = self.kernel_size

        cols, padded_shape = im2col(
            x.data, K, self.stride, self.padding
        )

        W_col = self.weight.data.reshape(self.out_channels, -1)

        out = cols @ W_col.T + self.bias.data

        H_p, W_p = padded_shape[2], padded_shape[3]

        out_h = (H_p - K) // self.stride + 1
        out_w = (W_p - K) // self.stride + 1

        out = out.transpose(0,2,1).reshape(N, self.out_channels, out_h, out_w)

        out_tensor = Tensor(out, requires_grad=True)
        # ---------------- backward
        def _backward():

            dout = out_tensor.grad.reshape(N, self.out_channels, -1).transpose(0, 2, 1)

            # gradients
            dW = np.zeros_like(W_col)
            dcols = np.zeros_like(cols)
            db = np.zeros_like(self.bias.data)

            for n in range(N):
                dW += dout[n].T @ cols[n]
                dcols[n] = dout[n] @ W_col

            db += dout.sum(axis=(0, 1))

            # reshape
            self.weight.grad += dW.reshape(self.weight.data.shape)
            self.bias.grad += db

            dx = col2im(dcols, x.data.shape, K)

            if x.requires_grad:
                x.grad += dx

        out_tensor._backward = _backward
        out_tensor._prev = {x}

        return out_tensor