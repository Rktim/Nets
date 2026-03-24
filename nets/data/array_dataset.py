import numpy as np


class ArrayDataset:

    def __init__(self, X, y):

        # Convert pandas → numpy
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        # Convert Tensor → numpy
        if hasattr(X, "data"):
            X = X.data

        if hasattr(y, "data"):
            y = y.data

        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]