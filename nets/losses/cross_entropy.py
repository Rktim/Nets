import numpy as np
from nets.tensor.tensor import Tensor


def cross_entropy(logits, target):
    """
    logits: (N, C)
    target: (N,)
    """

    # --- log-sum-exp trick
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)

    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    N = logits.data.shape[0]
    target_idx = target.data.astype(np.int32).reshape(-1)
    loss_val = -np.log(probs[np.arange(N), target_idx] + 1e-9).mean()

    out = Tensor(loss_val, requires_grad=True)

    def _backward():
        grad = probs.copy()
        grad[np.arange(N), target_idx] -= 1
        grad /= N

        logits.grad += grad

    out._backward = _backward
    out._prev = {logits}

    return out