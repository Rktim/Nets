import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# nets core
from nets.tensor.tensor import Tensor
from nets.nn import Sequential
from nets.nn.linear import Linear
from nets.nn.activations import ReLU
from nets.nn.conv2d import Conv2D
from nets.nn.maxpool2d import MaxPool2D
from nets.nn.flatten import Flatten

from nets.optim.sgd import SGD
from nets.losses.cross_entropy import cross_entropy

# data
from nets.data.array_dataset import ArrayDataset
from nets.data.dataloader import DataLoader

# trainer
from nets.trainer.trainer import Trainer

# visualization
from nets.visualization.logger import Logger
from nets.visualization.dashboard import Dashboard

# metrics
from nets.visualization.metrics import compute_confusion_matrix


# -------------------------
# LOAD DATA
# -------------------------
print("Loading MNIST...")

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(int)

# keep small (fast test)
X = X[:2000]
y = y[:2000]

X = X.reshape(-1, 1, 28, 28)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


# -------------------------
# DATA LOADERS
# -------------------------
train_loader = DataLoader(
    ArrayDataset(X_train, y_train),
    batch_size=128
)

val_loader = DataLoader(
    ArrayDataset(X_test, y_test),
    batch_size=128
)


# -------------------------
# MODEL
# -------------------------
model = Sequential(
    Conv2D(1, 8, 3, padding=1),
    ReLU(),
    MaxPool2D(2),

    Conv2D(8, 16, 3, padding=1),
    ReLU(),
    MaxPool2D(2),

    Flatten(),

    Linear(16 * 7 * 7, 64),
    ReLU(),

    Linear(64, 10)
)


# -------------------------
# LOGGER + RUN
# -------------------------
logger = Logger()

logger.load()  # load previous runs

logger.start_run("CNN MNIST (5 epochs test)")

logger.set_meta({
    "model": "CNN",
    "dataset": "MNIST",
    "epochs": 5
})


# -------------------------
# TRAINER
# -------------------------
optimizer = SGD(model.parameters(), lr=0.05)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=cross_entropy,
    logger=logger
)


# -------------------------
# TRAIN LOOP (CUSTOM LOGGING)
# -------------------------
EPOCHS = 5

for epoch in range(EPOCHS):

    # -------- TRAIN
    train_loss = 0
    train_acc = 0
    batches = 0

    for x, y in train_loader:

        logits = model(x)
        loss = cross_entropy(logits, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        preds = np.argmax(logits.data, axis=1)
        acc = (preds == y.data).mean()

        train_loss += loss.data
        train_acc += acc
        batches += 1

    train_loss /= batches
    train_acc /= batches

    logger.log({
        "type": "train",
        "epoch": epoch,
        "loss": float(train_loss),
        "accuracy": float(train_acc)
    })

    # -------- VALIDATION
    val_loss = 0
    val_acc = 0
    vb = 0

    all_preds = []
    all_targets = []

    for x, y in val_loader:

        logits = model(x)
        loss = cross_entropy(logits, y)

        preds = np.argmax(logits.data, axis=1)

        all_preds.append(logits.data)
        all_targets.append(y.data)

        val_loss += loss.data
        val_acc += (preds == y.data).mean()
        vb += 1

    val_loss /= vb
    val_acc /= vb

    logger.log({
        "type": "val",
        "epoch": epoch,
        "loss": float(val_loss),
        "accuracy": float(val_acc)
    })

    # -------- CONFUSION MATRIX
    preds_np = np.vstack(all_preds)
    targets_np = np.concatenate(all_targets).astype(np.int32).reshape(-1)
    cm = compute_confusion_matrix(preds_np, targets_np, 10)

    logger.log({
        "type": "confusion",
        "epoch": epoch,
        "cm": cm.tolist()
    })

    # -------- FEATURE MAPS (from first conv layer)
    for layer in model.layers:
        if hasattr(layer, "last_output"):
            logger.log({
                "type": "feature_maps",
                "epoch": epoch,
                "maps": layer.last_output[:1].tolist()
            })
            break


# -------------------------
# SAVE RUN
# -------------------------
logger.save()

print("✅ Training + Logging Complete")


# -------------------------
# LAUNCH DASHBOARD
# -------------------------
Dashboard(logger).run()