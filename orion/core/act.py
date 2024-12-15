import numpy as np

from orion.core import core

class ReLU(core.Function):
    def __init__(self, x: core.Node):
        super().__init__(input_nodes=[x])
        self.data = self.forward(x)

    def forward(self, x: np.array):
        return np.maximum(0, x.data)

    def _backward(self):
        grad = self.grad
        x = self.input_nodes[0]

        relu_grad = (self.data > 0).astype(np.float32)
        x.backward(grad * relu_grad)