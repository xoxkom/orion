# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

import numpy as np

from orion.tensor import Tensor

class DataLoader:
    """
    DataLoader: A utility class for batch loading datasets.

    This class takes a dataset (features and labels), splits it into batches,
    and optionally shuffles the data for each epoch. It supports iteration
    for training loops.

    Attributes:
        X (np.ndarray): Feature data of the dataset.
        y (np.ndarray): Label data of the dataset.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Flag indicating whether to shuffle the data.
        cursor (int): The current index for batch iteration.
        num_samples (int): Total number of samples in the dataset.
        indices (np.ndarray): Array of sample indices for shuffling.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Initialize the DataLoader with dataset, batch size, and shuffle option.

        :param dataset: A tuple containing features (X) and labels (y) as numpy arrays.
        :param batch_size: The number of samples in each batch.
        :param shuffle: A boolean flag indicating whether to shuffle the data each epoch.
        """
        self.X, self.y = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = None
        self.num_samples = self.X.shape[0]
        self.indices = np.arange(self.num_samples)
        self.reset()

    def reset(self):
        """
        Reset the DataLoader to the initial state.

        If shuffle is enabled, the dataset indices are shuffled.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cursor = 0

    def __iter__(self):
        """
        Make the DataLoader iterable.

        :return: Returns itself after resetting the state.
        """
        self.reset()
        return self

    def __next__(self):
        """
        Fetch the next batch of data.

        :return: A tuple of (batch_X, batch_y) as Tensor objects.
        :raises StopIteration: When all batches have been returned.
        """
        if self.cursor >= self.num_samples:
            raise StopIteration
        start = self.cursor
        end = min(start + self.batch_size, self.num_samples)
        self.cursor = end
        batch_X = self.X[self.indices[start:end]]
        batch_y = self.y[self.indices[start:end]]
        return Tensor(batch_X), Tensor(batch_y)

    def __len__(self):
        """
        Get the total number of batches.

        :return: The number of batches in the dataset.
        """
        return int(np.ceil(self.num_samples / self.batch_size))
