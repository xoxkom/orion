# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

import orion.nn as nn
from orion.core.graph import Graph
from orion.nn.loss import CrossEntropyLoss
from orion.tensor import Tensor


class Model(nn.Module):
    """
    A simple neural network model for classification tasks.

    The model consists of two fully connected (Linear) layers with a ReLU activation
    function in between. The input has 4 features, the hidden layer outputs 3 features,
    and the final output layer predicts probabilities for 2 classes.
    """

    def __init__(self):
        """
        Initialize the model with two Linear layers and a ReLU activation.

        - fc1: First Linear layer (4 -> 3)
        - relu: ReLU activation function
        - fc2: Second Linear layer (3 -> 2)
        """
        super().__init__()

        self.fc1 = nn.Linear(4, 3, initializer='he', label='fc1')  # First linear layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(3, 2, initializer='he', label='fc2')  # Output linear layer

    def forward(self, x):
        """
        Perform the forward pass of the model.

        :param x: Input tensor with shape (batch_size, 4).
        :return: Output tensor with shape (batch_size, 2), representing class logits.
        """
        x = self.fc1(x)  # First layer
        x = self.relu(x)  # Activation function
        x = self.fc2(x)  # Output layer
        return x


# Input data
x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # Two samples, each with 4 features

# Ground truth labels for classification
# Class indices: 0 for the first sample, 1 for the second sample
y = Tensor([0, 1])

# Instantiate the model
model = Model()

# Perform forward pass to get predictions
y_pred = model(x)

# Compute the cross-entropy loss
loss = CrossEntropyLoss(y_pred, y)

# Visualize the computation graph
# The computation graph includes the forward pass and loss calculation
# The resulting graph is saved to '../tmp/dot.png'
g = Graph()
g.view_graph(loss, '../tmp/dot.png')
