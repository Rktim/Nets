import matplotlib.pyplot as plt


def get_color(layer):

    name = layer.__class__.__name__

    if "Linear" in name:
        return "lightblue"
    elif "ReLU" in name:
        return "orange"
    elif "Sigmoid" in name:
        return "green"
    elif "Tanh" in name:
        return "purple"
    elif "Dropout" in name:
        return "red"
    else:
        return "gray"


def plot_model(model):

    if hasattr(model, "layers"):
        layers = model.layers
    else:
        layers = [model]

    layer_labels = []
    colors = []

    for layer in layers:

        if hasattr(layer, "weight"):
            in_f = layer.weight.data.shape[0]
            out_f = layer.weight.data.shape[1]
            label = f"{layer.__class__.__name__}\n({in_f} → {out_f})"
        else:
            label = layer.__class__.__name__

        layer_labels.append(label)
        colors.append(get_color(layer))

    plt.figure(figsize=(12, 3))

    for i, (label, color) in enumerate(zip(layer_labels, colors)):

        # node
        plt.text(
            i, 0, label,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="black")
        )

        # edge
        if i > 0:
            plt.plot([i - 1, i], [0, 0], 'k-')

    plt.title("Model Architecture (Color-coded)")
    plt.axis('off')
    plt.show()