import numpy as np
from orion.core.tensor import Tensor

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        DataLoader: 批量加载数据集
        :param dataset: (X, y) 数据集
        :param batch_size: 每个 batch 的大小
        :param shuffle: 是否打乱数据
        """
        self.X, self.y = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = None
        self.num_samples = self.X.shape[0]
        self.indices = np.arange(self.num_samples)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cursor = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.cursor >= self.num_samples:
            raise StopIteration
        start = self.cursor
        end = min(start + self.batch_size, self.num_samples)
        self.cursor = end
        batch_X = self.X[self.indices[start:end]]
        batch_y = self.y[self.indices[start:end]]
        return Tensor(batch_X), Tensor(batch_y)

    def __len__(self):
        """返回批次数"""
        return int(np.ceil(self.num_samples / self.batch_size))
