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


class ReLU(core.Function):
    """
    ReLU Activation Function Class

    Inherits from core.Function to perform the ReLU (Rectified Linear Unit)
    activation operation. It supports forward propagation and backward propagation.

    :param x: Input node (core.Node) for the ReLU activation.
    """

    def __init__(self, x: core.Node):
        """
        Initialize the ReLU activation function.

        :param x: Input node.
        """
        super().__init__(input_nodes=[x])
        self.data = self.forward(x)

    def forward(self, x: np.array):
        """
        Forward pass for ReLU activation.

        :param x: Input numpy array.
        :return: Output after applying ReLU (non-negative values).
        """
        return np.maximum(0, x.data)

    def _backward(self):
        """
        Backward pass for ReLU activation.

        Computes the gradient for backpropagation and propagates it to the input node.
        """
        grad = self.grad
        x = self.input_nodes[0]

        relu_grad = (self.data > 0).astype(np.float32)
        x.backward(grad * relu_grad)
