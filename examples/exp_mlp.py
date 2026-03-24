import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from nets.tensor.tensor import Tensor
from nets.nn import Sequential, Linear, ReLU
from nets.losses.cross_entropy import cross_entropy
from nets.optim.adam import Adam
from nets.visualization.logger import Logger
from nets.visualization.dashboard import Dashboard

# ---------- LOAD DATA
print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data / 255.0
y = mnist.target.astype(int)

# reduce for speed (still legit)
X = X[:15000]
y = y[:15000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------- MODEL (PROPER)
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)

# ---------- LOGGER
logger = Logger()
logger.start_run("MLP MNIST DEMO", meta={
    "task": "classification",
    "model": "MLP",
    "dataset": "MNIST"
})

# ---------- TRAIN (MINI-BATCH)
batch_size = 64
epochs = 20

for epoch in range(epochs):

    for i in range(0, len(X_train), batch_size):
        xb = Tensor(X_train[i:i+batch_size])
        yb = Tensor(y_train[i:i+batch_size])

        logits = model(xb)
        loss = cross_entropy(logits, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # ---------- EVAL
    logits = model(Tensor(X_test))
    preds = np.argmax(logits.data, axis=1)

    acc = (preds == y_test).mean()
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    logger.log({
        "epoch": epoch,
        "loss": float(loss.data),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    })

    if epoch % 5 == 0:
        cm = confusion_matrix(y_test, preds)
        logger.log({"type": "confusion", "cm": cm.tolist()})

Dashboard(logger).run()