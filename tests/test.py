from orion.core.act import ReLU
from orion.core.tensor import Tensor
from orion.core.layer import Linear

import numpy as np

# 输入张量
x = Tensor(np.array([[1.0, -2.0, 3.0]]))

# 创建线性层和 ReLU 激活
linear = Linear(3, 2)
relu = ReLU(linear.forward(x))

# 前向传播
print("ReLU Forward result:", relu.data)

# 反向传播
relu.backward(np.ones_like(relu.data))  # 反向传播
print("Gradients:")
print("linear.weight.grad:", linear.weight.grad)
print("linear.bias.grad:", linear.bias.grad)
