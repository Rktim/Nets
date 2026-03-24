import numpy as np
from nets.tensor.tensor import Tensor


def binary_cross_entropy(pred, target):

    eps = 1e-7
    pred_data = np.clip(pred.data, eps, 1 - eps)

    pred = Tensor(pred_data, requires_grad=pred.requires_grad)

    loss = -(target * pred.log() + (1 - target) * (1 - pred).log())

    return loss.mean()