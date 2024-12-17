# Orion

[English](README.md) | [简体中文](README_zh.md)

[![](https://img.shields.io/badge/Orion-Demo-brightgreen)](https://github.com/xoxkom/Orion) [![](https://img.shields.io/badge/Python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)

This is a simple deep learning framework based on a static computation graph. It references PyTorch's upper-layer interface and provides a straightforward lower-layer implementation for beginners to learn.

- **Computation Graph**: Supports static computation graph construction and visualization.
- **Automatic Differentiation**: Implements the automatic backpropagation mechanism.
- **Neural Network Modules**: Includes basic components such as linear layers and activation functions.
- **Loss Functions and Optimizers**: Provides implementations of Cross-Entropy Loss, Gradient Descent, etc.

## Project Structure

```angular2html
orion/
│
├── orion/                       # Core framework code
│   ├── core/                    # Core modules for computation graph and automatic differentiation
│   ├── nn/                      # Neural network modules
│   ├── data/                    # Batch data loader
│   └── tensor/                  # Tensor data structure
│
├── example/                     # Example code
│   ├── mnist.py                 # Example for training on the MNIST dataset
│   └── visual_graph.py          # Example for visualizing the computation graph
│
└── test/                        # Test scripts (TODO)
```

## Installation

If Python is not installed, please visit [![](https://img.shields.io/badge/Python-yellow)](https://www.python.org/) and download version 3.7 or above.

Clone the project:

```bash
git clone https://github.com/xoxkom/orion.git
cd orion
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Note: The computation graph visualization requires the installation of the [Graphviz](https://graphviz.org/download/) backend.

## Examples

### MNIST Handwritten Digit Classification Task Example:

Model Definition:

```python
import orion.nn as nn

class Model(nn.Module):
    """Fully connected neural network with two HE-initialized linear layers and ReLU activation"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128, initializer='he')
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, initializer='he')

    def forward(self, x):
        """Define forward propagation. nn.Module handles backpropagation logic and parameter registration."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Model()
```

Define Loss Function and Optimizer:

```python
from orion.nn.loss import CrossEntropyLoss
from orion.nn.optim import GD

optimizer = GD(model.parameters(), lr=0.1)
loss_fn = CrossEntropyLoss
```

Load Dataset:

```python
import pandas as pd

from orion.data import DataLoader

def load_mnist_csv(train_path, test_path):
    # Ensure the dataset exists
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values / 255.0  # Normalize pixel values
    y_train = train_data.iloc[:, 0].values          # Labels

    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values

    return (X_train, y_train), (X_test, y_test)

# Load dataset
train_path = "../datasets/mnist_train.csv"  # Set actual path
test_path = "../datasets/mnist_test.csv"
(X_train, y_train), (X_test, y_test) = load_mnist_csv(train_path, test_path)

# Create DataLoader for batching
batch_size = 32
train_loader = DataLoader((X_train, y_train), batch_size=batch_size)
test_loader = DataLoader((X_test, y_test), batch_size=batch_size)
```

Train the Model:

```python
def train_model(model, dataloader, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass and optimizer update
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.data

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

train_model(model, train_loader, optimizer, loss_fn, epochs=20)
```

Test the Model:

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

### Computation Graph Visualization:

```python
import orion.nn as nn
from orion.core.graph import Graph
from orion.nn.loss import CrossEntropyLoss
from orion.tensor import Tensor

# Model definition
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 3, initializer='he', label='fc1')  # First linear layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(3, 2, initializer='he', label='fc2')  # Output linear layer

    def forward(self, x):
        x = self.fc1(x)  # First layer
        x = self.relu(x)  # Activation function
        x = self.fc2(x)  # Output layer
        return x


# Sample input and output data
x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # Two samples, each with 4 features
y = Tensor([0, 1])

model = Model()

# Forward propagation
y_pred = model(x)

# Define loss function
loss = CrossEntropyLoss(y_pred, y)

# Instantiate Graph and call visualization
# Requires Graphviz backend installation
g = Graph()
g.view_graph(loss, '../tmp/dot.png')
```