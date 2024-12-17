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


class Loss(core.Function):
    """
    Base class for all loss functions in the computation graph.

    Inherits from core.Function and provides a framework for implementing
    custom loss functions with forward and backward propagation.

    Attributes:
        y_pred (core.Node): The predicted output from the model.
        y_true (core.Node): The ground truth labels.
        data (float): The computed loss value after the forward pass.
    """

    def __init__(self, y_pred: core.Node, y_true: core.Node):
        """
        Initialize the Loss function.

        :param y_pred: Predicted output from the model.
        :param y_true: Ground truth labels.
        """
        super().__init__(input_nodes=[y_pred, y_true])
        self.y_pred = y_pred
        self.y_true = y_true
        self.data = self.forward(y_pred.data, y_true.data)

    def forward(self, y_pred, y_true):
        """
        Compute the forward pass for the loss function.

        :param y_pred: Predicted output.
        :param y_true: Ground truth labels.
        :raises NotImplementedError: Must be implemented in the derived class.
        """
        raise NotImplementedError

    def _backward(self):
        """
        Compute the backward pass for the loss function.

        :raises NotImplementedError: Must be implemented in the derived class.
        """
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss: Computes the cross-entropy loss for classification tasks.

    Cross-entropy loss measures the performance of a classification model whose
    output is a probability distribution.

    Attributes:
        probs (np.ndarray): Softmax probabilities for the predicted logits.
    """

    def __init__(self, y_pred: core.Node, y_true: core.Node):
        """
        Initialize the CrossEntropyLoss.

        :param y_pred: Predicted logits from the model.
        :param y_true: Ground truth labels.
        """
        super().__init__(y_pred, y_true)
        self.probs = None
        self.data = self.forward(y_pred.data, y_true.data)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Compute the forward pass for the cross-entropy loss.

        :param y_pred: Predicted logits (before applying softmax).
        :param y_true: Ground truth labels.
        :return: Computed loss value.
        """
        exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))  # For numerical stability
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Softmax probabilities
        self.y_true = y_true

        batch_size = y_pred.shape[0]
        log_probs = -np.log(self.probs[range(batch_size), y_true] + 1e-12)
        return np.sum(log_probs) / batch_size

    def _backward(self):
        """
        Compute the backward pass for the cross-entropy loss.

        Calculates the gradient of the loss with respect to the logits and propagates
        it backward through the computation graph.
        """
        logits = self.input_nodes[0]
        batch_size = self.probs.shape[0]

        grad = self.probs.copy()
        grad[range(batch_size), self.y_true] -= 1  # Subtract 1 from the true class probabilities
        grad /= batch_size  # Normalize by batch size

        logits.backward(grad)