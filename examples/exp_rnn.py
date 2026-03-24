import numpy as np
from sklearn.metrics import accuracy_score

from nets.tensor.tensor import Tensor
from nets.nn import Sequential, Linear, RNN
from nets.losses.cross_entropy import cross_entropy
from nets.optim.adam import Adam
from nets.visualization.logger import Logger
from nets.visualization.dashboard import Dashboard

# ---------- DATA
np.random.seed(42)

X = np.random.randn(1000, 5, 3)
y = (X.sum(axis=(1, 2)) > 0).astype(int)

# ---------- MODEL
model = Sequential(
    RNN(input_size=3, hidden_size=16),
    Linear(16, 2)
)

optimizer = Adam(model.parameters(), lr=0.001)

# ---------- LOGGER
logger = Logger()
logger.start_run("RNN Demo", meta={
    "task": "classification",
    "model": "RNN",
    "dataset": "Sequence"
})

# ---------- TRAIN
for epoch in range(20):

    logits = model(Tensor(X))
    loss = cross_entropy(logits, Tensor(y))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    preds = np.argmax(logits.data, axis=1)
    acc = accuracy_score(y, preds)

    logger.log({
        "epoch": epoch,
        "loss": float(loss.data),
        "accuracy": float(acc)
    })

Dashboard(logger).run()