import numpy as np

from orion.core import core, layers

input_data = core.Tensor(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
fc1 = layers.Linear(in_features=4, out_features=2, initializer="default", label="fc1")
fc2 = layers.Linear(in_features=2, out_features=5, initializer="default", label="fc2")
fc3 = layers.Linear(in_features=5, out_features=1, initializer="default", label="fc3")

y = fc1.forward(input_data)
y2 = fc2.forward(y)
y3 = fc3.forward(y2)

print(f"fc1 Weight: {fc1.weight}")
print(f"fc1 Bias: {fc1.bias}")
print(f"fc1 Output: {y}")

print(f"fc2 Weight: {fc2.weight}")
print(f"fc2 Bias: {fc2.bias}")
print(f"fc2 Output: {y2}")

core.view_graph(y3, save_path=f"../tmp/dot.png")