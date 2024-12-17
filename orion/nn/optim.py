# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

from abc import ABC, abstractmethod

from orion.core.graph import Graph


class Optimizer(ABC):
    """
    Base class for all optimizers.

    Optimizers are responsible for updating the model parameters based on gradients
    computed during backpropagation. This base class provides a framework for
    implementing custom optimizers.

    Attributes:
        parameters (list): List of model parameters to optimize.
        lr (float): Learning rate for the optimization step.
    """

    def __init__(self, parameters, lr):
        """
        Initialize the optimizer.

        :param parameters: List of model parameters to update.
        :param lr: Learning rate for parameter updates.
        """
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def _step(self):
        """
        Abstract method to implement parameter update logic in derived classes.
        """
        pass

    def step(self):
        """
        Perform an optimization step and clear gradients.

        This method calls the `_step` method for updating parameters, clears gradients,
        and resets the computation graph.
        """
        self._step()
        self.zero_grad()

        g = Graph()
        g.clear()

    def zero_grad(self):
        """
        Reset gradients for all parameters to zero.
        """
        for param in self.parameters:
            param.zero_grad()


class GD(Optimizer):
    """
    Gradient Descent (GD) optimizer.

    Updates parameters by subtracting the product of the learning rate and gradients:
        param.data -= lr * param.grad

    Attributes:
        parameters (list): List of model parameters to update.
        lr (float): Learning rate for the optimization step.
    """

    def __init__(self, parameters: list, lr: float = 0.01):
        """
        Initialize the GD optimizer.

        :param parameters: List of model parameters to update.
        :param lr: Learning rate for parameter updates (default: 0.01).
        """
        super().__init__(parameters, lr)

    def _step(self):
        """
        Perform the parameter update step for GD.

        :raises ValueError: If a parameter has no gradient computed.
        """
        for param in self.parameters:
            if param.grad is None:
                raise ValueError(f"Parameter {param.label} has no gradient. Did you forget to call backward?")
            param.data -= self.lr * param.grad
