import numpy as np
from nets.data.array_dataset import ArrayDataset
from nets.data.dataloader import DataLoader


X = np.random.randn(100, 10)
y = np.random.randint(0, 2, size=(100,))

dataset = ArrayDataset(X, y)
loader = DataLoader(dataset, batch_size=16)

for batch_x, batch_y in loader:
    print(batch_x.data.shape, batch_y.data.shape)