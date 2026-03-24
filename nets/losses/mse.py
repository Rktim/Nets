def mse(pred, target):
    diff = pred - target
    return (diff * diff).mean()