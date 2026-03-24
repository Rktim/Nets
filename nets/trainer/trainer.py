from nets.metrics import accuracy
from nets.tensor.tensor import no_grad
from nets.optim.utils import clip_gradients


class Trainer:

    def __init__(self, model, optimizer, loss_fn, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = logger

    def fit(self, train_loader, val_loader=None, epochs=10):

        for epoch in range(epochs):

            # -------- TRAIN --------
            total_loss = 0
            total_acc = 0
            batches = 0

            for x, y in train_loader:

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                loss.backward()

                # gradient clipping
                clip_gradients(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                acc = accuracy(logits, y)

                total_loss += loss.data
                total_acc += acc
                batches += 1

            train_loss = total_loss / batches
            train_acc = total_acc / batches

            # -------- VALIDATION --------
            val_loss, val_acc = None, None

            if val_loader is not None:

                v_loss = 0
                v_acc = 0
                v_batches = 0

                with no_grad():
                    for x, y in val_loader:
                        logits = self.model(x)
                        loss = self.loss_fn(logits, y)

                        acc = accuracy(logits, y)

                        v_loss += loss.data
                        v_acc += acc
                        v_batches += 1

                val_loss = v_loss / v_batches
                val_acc = v_acc / v_batches

            # -------- LOGGING --------
            if self.logger:
                self.logger.log({
                "type": "train",
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_acc,
                "lr": self.optimizer.lr
            })
                if val_loader:
                    self.logger.log({
                    "type": "val",
                    "epoch": epoch,
                    "loss": val_loss,
                    "accuracy": val_acc
                })
                            # -------- PRINT --------
            if val_loader:
                print(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}"
                )