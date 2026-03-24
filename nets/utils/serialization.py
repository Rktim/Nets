import numpy as np


def state_dict(model):
    sd = {}
    for i, p in enumerate(model.parameters()):
        sd[f"param_{i}"] = p.data.copy()
    return sd


def load_state_dict(model, sd):
    for i, p in enumerate(model.parameters()):
        key = f"param_{i}"
        if key in sd:
            p.data[...] = sd[key]
        else:
            raise KeyError(f"Missing key {key} in state_dict")


def save_model(model, path):
    sd = state_dict(model)
    np.savez(path, **sd)


def load_model(model, path):
    data = np.load(path)
    sd = {k: data[k] for k in data.files}
    load_state_dict(model, sd)