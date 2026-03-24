import numpy as np


def clip_gradients(parameters, max_norm=1.0):

    total_norm = 0.0

    for p in parameters:
        if p.grad is not None:
            total_norm += np.sum(p.grad ** 2)

    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)

        for p in parameters:
            if p.grad is not None:
                p.grad *= scale