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
import orion.nn as nn
from orion.core import act

class Linear(nn.Module):
    """
    Linear Layer: Fully connected layer for neural networks.

    This layer applies a linear transformation to the input data:
        y = x @ weight + bias

    Attributes:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        weight (Parameter): Learnable weight matrix.
        bias (Parameter): Learnable bias vector (optional).
        label (str): Layer label for identification.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 initializer: str = "default",
                 label: str = None):
        """
        Initialize the Linear layer with weights and optional bias.

        :param in_features: Number of input features.
        :param out_features: Number of output features.
        :param bias: Whether to include a bias term (default: True).
        :param initializer: Weight initialization method.
        :param label: Optional label for the layer.
        """
        super().__init__()
        if label is not None:
            self.label = label
        else:
            self.label = "Linear"

        self.weight = core.Parameter(shape=(in_features, out_features), initializer=initializer, label=f"{self.label}.weight")
        self.bias = core.Parameter(shape=(out_features,), initializer="default", label=f"{self.label}.bias") if bias else None

        self.in_features = in_features
        self.out_features = out_features
        self._parameters = [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward(self, x: core.Node):
        """
        Perform the forward pass for the Linear layer.

        :param x: Input tensor of shape (batch_size, in_features).
        :return: Output tensor of shape (batch_size, out_features).
        :raises ValueError: If input shape does not match in_features.
        """
        if x.shape[1] != self.in_features:
            raise ValueError(f"Input shape {x.shape[1]} does not match in_features {self.in_features}")

        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

class ReLU(nn.Module):
    """
    ReLU Activation Layer: Applies the ReLU activation function element-wise.

    ReLU (Rectified Linear Unit) is defined as:
        ReLU(x) = max(0, x)

    Attributes:
        label (str): Layer label for identification.
    """
    def __init__(self, label: str = None):
        """
        Initialize the ReLU activation layer.

        :param label: Optional label for the layer.
        """
        super().__init__()
        if label is not None:
            self.label = label
        else:
            self.label = "ReLU"

    def forward(self, x: core.Node):
        """
        Perform the forward pass for the ReLU activation.

        :param x: Input tensor.
        :return: Output tensor with ReLU applied element-wise.
        """
        return act.ReLU(x)
