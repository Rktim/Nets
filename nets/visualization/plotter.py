import matplotlib.pyplot as plt


def plot_loss(history):

    epochs = history["epoch"]
    loss = history["loss"]

    plt.figure()
    plt.plot(epochs, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()

    plt.show()