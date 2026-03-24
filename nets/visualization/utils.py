def ema(values, alpha=0.1):
    if not values:
        return []

    smoothed = []
    avg = values[0]

    for v in values:
        avg = alpha * v + (1 - alpha) * avg
        smoothed.append(avg)

    return smoothed