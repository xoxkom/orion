# Orion

[English](README.md) | [简体中文](README_zh.md)

[![](https://img.shields.io/badge/Orion-Demo-brightgreen)](https://github.com/xoxkom/Orion) [![](https://img.shields.io/badge/Python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)

这是一个非常简单的深度学习框架，基于静态计算图，参考PyTorch的上层接口，提供了简单的底层实现，供初学者学习。

- **计算图**：支持静态计算图的构建与可视化。
- **自动求导**：实现反向传播自动求导机制。
- **神经网络模块**：包含线性层、激活函数等基础组件。
- **损失函数与优化器**：提供交叉熵损失、梯度下降等实现。

## 项目结构

```angular2html
orion/
│
├── orion/                       # 框架核心代码
│   ├── core/                    # 计算图与自动求导核心模块
│   ├── nn/                      # 神经网络模块
│   ├── data/                    # 批次数据加载器
│   └── tensor/                  # Tensor 数据结构
│
├── example/                     # 示例代码
│   ├── mnist.py                 # MNIST 手写数字数据集训练的示例
│   └── visual_graph.py          # 可视化计算图的示例
│
└── test/                        # 测试脚本（TODO）
```

## 安装

如果你没有安装 Python，请移步 [![](https://img.shields.io/badge/Python-yellow)](https://www.python.org/) ，并下载3.7及以上版本。

克隆项目

```bash
git clone https://github.com/xoxkom/orion.git
cd orion
```

安装依赖

```bash
pip install -r requirements.txt
```

**注意**：计算图可视化需要安装[Graphviz](https://graphviz.org/download/)后端.

## 示例

### MNIST手写数字数据集分类任务示例：

模型定义：

```python
import orion.nn as nn

class Model(nn.Module):
    """ 全连接神经网络，包括两个he初始化权重的全连接层，使用ReLU激活 """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128, initializer='he')
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, initializer='he')

    def forward(self, x):
        """ 定义模型的前向传播，nn.Module会自动处理反向传播逻辑和参数注册逻辑 """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Model()
```

定义损失函数和优化器：

```python
from orion.nn.loss import CrossEntropyLoss
from orion.nn.optim import GD

optimizer = GD(model.parameters(), lr=0.1)
loss_fn = CrossEntropyLoss
```

加载数据集：
```python
import pandas as pd

from orion.data import DataLoader

def load_mnist_csv(train_path, test_path):
    # 请确保数据集存在
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values

    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values

    return (X_train, y_train), (X_test, y_test)

# 加载数据集
train_path = "../datasets/mnist_train.csv"  # 请根据实际路径设置
test_path = "../datasets/mnist_test.csv"
(X_train, y_train), (X_test, y_test) = load_mnist_csv(train_path, test_path)

# 创建 DataLoader 实例，用于批处理
batch_size = 32
train_loader = DataLoader((X_train, y_train), batch_size=batch_size)
test_loader = DataLoader((X_test, y_test), batch_size=batch_size)
```

训练模型：

```python
def train_model(model, dataloader, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            # 前向传播
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # 反向传播、优化器更新
            loss.backward()
            optimizer.step()

            # 累加损失
            total_loss += loss.data

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

train_model(model, train_loader, optimizer, loss_fn, epochs=20)
```

测试模型：

```python
import numpy as np

def test_model(model, test_loader):
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        predictions = np.argmax(y_pred.data, axis=1)
        correct += np.sum(predictions == y_batch.data)
        total += y_batch.shape[0]

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

test_model(model, test_loader)
```

### 计算图可视化：

```python
import orion.nn as nn
from orion.core.graph import Graph
from orion.nn.loss import CrossEntropyLoss
from orion.tensor import Tensor

# 模型定义
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 3, initializer='he', label='fc1')
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, initializer='he', label='fc2')

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 示例输入、输出数据
x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
y = Tensor([0, 1])

model = Model()

# 前向传播
y_pred = model(x)

# 定义损失函数
loss = CrossEntropyLoss(y_pred, y)

# 实例化Graph实例，并调用绘图方法
# 调用该方法需要安装Graphviz后端
g = Graph()
g.view_graph(loss, '../tmp/dot.png')
```