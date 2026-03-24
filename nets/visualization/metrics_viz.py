import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm):

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.colorbar()

    # annotate values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()
    
    
def plot_confusion_heatmap(cm):

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, cmap="viridis")

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     ha="center", va="center", color="white")

    plt.tight_layout()
    plt.show()
    
    
def plot_roc_curve(logits, y_true, num_classes):

    from itertools import cycle

    y = y_true.data
    scores = logits.data

    plt.figure()

    colors = cycle(['blue', 'red', 'green', 'orange', 'purple',
                    'brown', 'pink', 'gray', 'olive', 'cyan'])

    for i, color in zip(range(num_classes), colors):

        # One-vs-rest
        y_binary = (y == i).astype(int)
        score = scores[:, i]

        # sort
        sorted_idx = np.argsort(score)
        y_binary = y_binary[sorted_idx]

        tpr = []
        fpr = []

        P = np.sum(y_binary == 1)
        N = np.sum(y_binary == 0)

        tp = 0
        fp = 0

        for j in range(len(y_binary)-1, -1, -1):
            if y_binary[j] == 1:
                tp += 1
            else:
                fp += 1

            tpr.append(tp / (P + 1e-9))
            fpr.append(fp / (N + 1e-9))

        plt.plot(fpr, tpr, color=color, label=f"Class {i}")

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend()

    plt.show()
    
    
def plot_training_curves(history):

    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"])
    plt.title("Loss")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"])
    plt.title("Accuracy")

    plt.show()