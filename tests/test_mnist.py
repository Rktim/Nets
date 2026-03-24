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
from nets.metrics import accuracy, confusion_matrix


# -------------------------
# LOAD DATA
# -------------------------
print("Loading MNIST...")

mnist = fetch_openml('mnist_784', version=1)

# FIX: convert to numpy
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(int)

# reduce for CPU speed
X = X[:10000]
y = y[:10000]

# reshape for CNN
X = X.reshape(-1, 1, 28, 28)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


# -------------------------
# DATALOADER
# -------------------------
train_dataset = ArrayDataset(X_train, y_train)
val_dataset = ArrayDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)


# -------------------------
# CNN MODEL
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
# TRAINING SETUP
# -------------------------
optimizer = SGD(model.parameters(), lr=0.05)
logger = Logger()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=cross_entropy,
    logger=logger
)


# -------------------------
# TRAIN (NO THREADING - DEBUG SAFE)
# -------------------------
print("\n🚀 Training started...\n")

try:
    trainer.fit(train_loader, val_loader, epochs=15)

    print("\n✅ Training Completed")

except Exception as e:
    import traceback
    print("\n🔥 TRAINING ERROR:")
    traceback.print_exc()


# -------------------------
# FINAL EVALUATION
# -------------------------
print("\n📊 Evaluating on test set...")

x_test_tensor = Tensor(X_test)
y_test_tensor = Tensor(y_test)

logits = model(x_test_tensor)

acc = accuracy(logits, y_test_tensor)

print(f"\n🎯 FINAL TEST ACCURACY: {acc:.4f}")

cm = confusion_matrix(logits, y_test_tensor, num_classes=10)
print("\nConfusion Matrix:\n", cm)


# -------------------------
# DASHBOARD (AFTER TRAINING)
# -------------------------
print("\n📈 Launching Dashboard...")
Dashboard(logger).run()