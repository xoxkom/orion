import numpy as np

from orion.core import core

class SGD:
    def __init__(self, parameters: list, lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is None:
                raise ValueError(f"Parameter {param.label} has no gradient. Did you forget to call backward?")
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
