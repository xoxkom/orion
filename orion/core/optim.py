import numpy as np

from orion.core import core

class SGD(object):
    def __init__(self, parameters: list[core.Parameter], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is None:
                raise ValueError(f"Parameter {param.label} has no gradient. Did you forget to call backward?")
            param.data -= self.lr * param.grad