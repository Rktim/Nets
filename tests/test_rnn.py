import numpy as np

from nets.tensor.tensor import Tensor
from nets.nn import Sequential
from nets.nn.linear import Linear
from nets.nn.activations import ReLU
from nets.nn.rnn import RNN

from nets.optim.sgd import SGD
from nets.losses.cross_entropy import cross_entropy

from nets.data.array_dataset import ArrayDataset
from nets.data.dataloader import DataLoader

from nets.visualization.logger import Logger
from nets.visualization.dashboard import Dashboard


# ---------------- DATA ----------------
def generate_data(n=2000, seq_len=10):

    X = np.random.randn(n, seq_len, 1)

    # EASY + STABLE TASK
    y = (X[:, -1, 0] > 0).astype(int)

    # normalization (VERY IMPORTANT)
    X = (X - X.mean()) / (X.std() + 1e-8)

    return X.astype(np.float32), y.astype(np.int32)


X, y = generate_data()

# split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]


# ---------------- DATALOADER ----------------
train_loader = DataLoader(
    ArrayDataset(X_train, y_train),
    batch_size=64
)

val_loader = DataLoader(
    ArrayDataset(X_val, y_val),
    batch_size=64
)


# ---------------- MODEL ----------------
model = Sequential(
    RNN(1, 32),
    Linear(32, 16),
    ReLU(),
    Linear(16, 2)
)


# ---------------- LOGGER ----------------
logger = Logger()
logger.load()

logger.start_run("RNN Stable Test")

logger.set_meta({
    "model": "RNN",
    "dataset": "Sequence (Last Step)",
    "epochs": 10
})


# ---------------- OPTIMIZER ----------------
optimizer = SGD(model.parameters(), lr=0.01)


# ---------------- TRAIN LOOP ----------------
EPOCHS = 10

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    train_loss = 0
    train_acc = 0
    batches = 0

    for x, y in train_loader:

        logits = model(x)
        loss = cross_entropy(logits, y)

        loss.backward()

        # ---- GRADIENT CLIPPING (CRITICAL)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = np.clip(p.grad, -1.0, 1.0)

        optimizer.step()
        optimizer.zero_grad()

        preds = np.argmax(logits.data, axis=1)
        acc = (preds == y.data).mean()

        train_loss += loss.data
        train_acc += acc
        batches += 1

    train_loss /= batches
    train_acc /= batches

    # ---- LOG TRAIN
    logger.log({
        "type": "train",
        "epoch": epoch,
        "loss": float(train_loss),
        "accuracy": float(train_acc)
    })


    # ---- VALIDATION ----
    val_loss = 0
    val_acc = 0
    vb = 0

    for x, y in val_loader:

        logits = model(x)
        loss = cross_entropy(logits, y)

        preds = np.argmax(logits.data, axis=1)
        acc = (preds == y.data).mean()

        val_loss += loss.data
        val_acc += acc
        vb += 1

    val_loss /= vb
    val_acc /= vb

    # ---- LOG VAL
    logger.log({
        "type": "val",
        "epoch": epoch,
        "loss": float(val_loss),
        "accuracy": float(val_acc)
    })

    # ---- PRINT (OPTIONAL DEBUG)
    print(f"Epoch {epoch} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")


# ---------------- SAVE RUN ----------------
logger.save()

print("✅ Training + Logging Complete")


# ---------------- DASHBOARD ----------------
Dashboard(logger).run()