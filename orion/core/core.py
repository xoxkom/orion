# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
# 
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

from __future__ import annotations

import weakref
from abc import abstractmethod

import numpy as np

from orion.core.graph import Graph

__all__ = [
    "Node",
    "Function",
    "Parameter"
]

_default_graph = Graph()


class Node(object):
    """
    Base class for all types of nodes in the static computation graph.

    Attributes:
        data (np.ndarray): Node's data value.
        grad (np.ndarray): Gradient value for backpropagation.
        label (str): Label name of the node.
        next_nodes (weakref.WeakSet): References to subsequent nodes in the graph.
    """
    _global_id_counter = {}

    def __init__(self):
        self.next_nodes = weakref.WeakSet()
        self.data = None
        self.grad = None
        self.label = self.__class__.__name__

        class_name = self.__class__.__name__
        if class_name not in Node._global_id_counter:
            Node._global_id_counter[class_name] = 0
        Node._global_id_counter[class_name] += 1
        self.id = f"{class_name}_{Node._global_id_counter[class_name]}"

        global _default_graph
        _default_graph.append(self)

    def backward(self, grad: np.ndarray = None):
        """
        Perform backward propagation to compute gradients.

        :param grad: Gradient from the subsequent node.
        """
        if grad is None and self.grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if isinstance(self, Function):
            self._backward()

    def __neg__(self):
        return Negative(self)

    def __add__(self, other: Node):
        return Add(self, other)

    def __sub__(self, other: Node):
        return Minus(self, other)

    def __mul__(self, other: Node):
        return Multiply(self, other)

    def __matmul__(self, other: Node):
        return Matmul(self, other)

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.data)})"

    def __repr__(self):
        return str(self)

    @property
    def numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return np.array(self.data)

    @property
    def shape(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return np.array(self.data).shape

    def to_numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return np.array(self.data)

    def to_list(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return list(self.data)


class Function(Node):
    """
    Base class for all functions in the computation graph.

    Inherits from Node and represents an operation in the graph.

    Attributes:
        _input_nodes (list): Weak references to input nodes.
    """

    def __init__(self, input_nodes: list[Node]):
        super().__init__()
        self._input_nodes = [weakref.ref(node) for node in input_nodes]
        for node in input_nodes:
            node.next_nodes.add(self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def _backward(self, *args, **kwargs):
        pass

    @property
    def input_nodes(self):
        """
        Retrieve the input nodes by dereferencing weak references.
        :return: List of input nodes.
        """
        return [node_ref() for node_ref in self._input_nodes if node_ref() is not None]


class Parameter(Node):
    """
    Parameter node representing trainable weights.

    Attributes:
        grad (np.ndarray): Gradient associated with the parameter.
    """

    def __init__(self,
                 init_value: np.ndarray | list = None,
                 shape: tuple[int, ...] = None,
                 initializer: str = "default",
                 label: str = None):
        super().__init__()
        if init_value is not None:
            self.data = np.array(init_value, dtype=np.float32)
        elif shape is not None:
            if initializer == "default":
                self.data = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
            elif initializer == "he":
                fan_in = shape[0]
                std = np.sqrt(2 / fan_in)
                self.data = np.random.randn(*shape).astype(np.float32) * std
            elif initializer == "xavier":
                fan_in, fan_out = shape[0], shape[1]
                std = np.sqrt(2 / (fan_in + fan_out))
                self.data = np.random.randn(*shape).astype(np.float32) * std
            else:
                raise ValueError(f"Unsupported initializer: {initializer}")
        else:
            raise ValueError("Either `init_value` or `shape` must be provided.")

        self.grad = None
        if label is not None:
            self.label = label

    def zero_grad(self):
        """Reset the gradient to None."""
        self.grad = None


class Add(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x + y

    def _backward(self):
        grad = self.grad
        x, y = self.input_nodes
        x.backward(grad)
        if y.shape != grad.shape:
            y.backward(np.sum(grad, axis=tuple(range(len(grad.shape) - len(y.shape)))))
        else:
            y.backward(grad)


class Negative(Function):
    def __init__(self, x: Node):
        super().__init__(input_nodes=[x])
        self.data = self.forward(x.data)

    def forward(self, x: np.ndarray):
        return -1. * x

    def _backward(self):
        grad = self.grad
        self.input_nodes[0].backward(-grad)


class Minus(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x - y

    def _backward(self):
        grad = self.grad
        self.input_nodes[0].backward(grad)
        self.input_nodes[1].backward(-grad)


class Multiply(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x * y

    def _backward(self):
        grad = self.grad
        x, y = self.input_nodes
        x.backward(grad * y.data)
        y.backward(grad * x.data)


class Matmul(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x @ y

    def _backward(self):
        grad = self.grad
        x, y = self.input_nodes
        x.backward(grad @ y.data.T)
        y.backward(x.data.T @ grad)
