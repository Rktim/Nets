import numpy as np
from nets.tensor.tensor import Tensor


class DataLoader:

    def __init__(self, dataset, batch_size=32, shuffle=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(dataset))

    def __iter__(self):

        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current = 0
        return self

    def __next__(self):

        if self.current >= len(self.indices):
            raise StopIteration

        batch_idx = self.indices[
            self.current:self.current + self.batch_size
        ]

        batch_X = []
        batch_y = []

        for idx in batch_idx:
            x, y = self.dataset[idx]
            batch_X.append(x)
            batch_y.append(y)

        self.current += self.batch_size
        batch_X = self.dataset.X[batch_idx]
        batch_y = self.dataset.y[batch_idx]

        batch_X = Tensor(batch_X.astype(np.float32))
        batch_y = Tensor(batch_y.astype(np.int32))
        return batch_X, batch_y