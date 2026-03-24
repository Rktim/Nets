import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

from nets.tensor.tensor import Tensor
from nets.nn import Sequential, Linear, ReLU
from nets.losses.cross_entropy import cross_entropy
from nets.optim.adam import Adam
from nets.visualization.logger import Logger
from nets.visualization.dashboard import Dashboard

# ---------- DATA
digits = load_digits()
X = digits.data / 16.0
y = digits.target

# ---------- MODEL
model = Sequential(
    Linear(64, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)

# ---------- LOGGER
logger = Logger()
logger.start_run("Digits Demo", meta={
    "task": "classification",
    "model": "CNN",
    "dataset": "Digits"
})

# ---------- TRAIN
for epoch in range(20):

    logits = model(Tensor(X))
    loss = cross_entropy(logits, Tensor(y))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    preds = np.argmax(logits.data, axis=1)
    acc = (preds == y).mean()

    logger.log({
        "epoch": epoch,
        "loss": float(loss.data),
        "accuracy": float(acc)
    })

    if epoch % 5 == 0:
        cm = confusion_matrix(y, preds)
        logger.log({"type": "confusion", "cm": cm.tolist()})

Dashboard(logger).run()