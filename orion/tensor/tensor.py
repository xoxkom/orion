# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

import numpy as np

import orion.core as core

class Tensor(core.Node):
    """
    Tensor class for managing multi-dimensional data.

    Inherits from core.Node and acts as the primary data structure
    for storing and processing numerical arrays within the computation graph.

    Attributes:
        data (np.ndarray): The numerical data represented as a NumPy array.
    """
    def __init__(self, data):
        """
        Initialize a Tensor with input data.

        :param data: A NumPy ndarray containing numerical data.
        :raises TypeError: If the input data is not a NumPy ndarray.
        """
        super().__init__()

        self.data = np.array(data)

    @property
    def shape(self):
        """
        Return the shape of the tensor.

        :return: Tuple representing the shape of the tensor data.
        """
        return self.data.shape

    def __str__(self):
        """
        Return a string representation of the Tensor.

        :return: A string describing the shape and data of the tensor.
        """
        return f"Tensor(shape={self.shape}, data={self.data})"
