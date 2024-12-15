import numpy as np

from orion.core import core

class Tensor(core.Node):
    def __init__(self, data: np.ndarray):
        super().__init__()
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy ndarray.")

        self.data = data

    @property
    def shape(self):
        """Return the shape of the tensor."""
        return self.data.shape

    def __str__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"