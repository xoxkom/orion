import pandas as pd
import numpy as np
from orion.core.loss import CrossEntropyLoss
from orion.core.optim import SGD
from orion.core.layer import Linear, ReLU
from orion.data import DataLoader

# 加载 MNIST 数据集（CSV 格式）
def load_mnist_csv(train_path, test_path):
    """
    使用 pandas 加载 MNIST 数据集
    :param train_path: 训练集 CSV 文件路径
    :param test_path: 测试集 CSV 文件路径
    :return: (X_train, y_train), (X_test, y_test)
    """
    # 读取训练集
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values / 255.0  # 归一化特征
    y_train = train_data.iloc[:, 0].values          # 标签

    # 读取测试集
    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values

    return (X_train, y_train), (X_test, y_test)

# 定义模型
class SimpleModel:
    def __init__(self):
        self.fc1 = Linear(784, 128, initializer='he', label='fc1')
        self.relu1 = ReLU(label='relu1')
        self.fc2 = Linear(128, 10, initializer='he', label='fc2')

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        return x

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

# 训练过程
def train_model(model, dataloader, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            # 前向传播
            y_pred = model.forward(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 累计损失
            total_loss += loss.data

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# 测试过程
def test_model(model, dataloader):
    correct = 0
    total = 0
    for X_batch, y_batch in dataloader:
        y_pred = model.forward(X_batch)
        predictions = np.argmax(y_pred.data, axis=1)
        correct += np.sum(predictions == y_batch.data)
        total += y_batch.shape[0]

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# 主程序
if __name__ == "__main__":
    # 数据路径
    train_path = "../datasets/mnist_train.csv"
    test_path = "../datasets/mnist_test.csv"

    # 加载数据集
    (X_train, y_train), (X_test, y_test) = load_mnist_csv(train_path, test_path)

    # 创建 DataLoader
    batch_size = 128
    train_loader = DataLoader((X_train, y_train), batch_size=batch_size)
    test_loader = DataLoader((X_test, y_test), batch_size=batch_size)

    # 初始化模型、损失函数和优化器
    model = SimpleModel()
    loss_fn = CrossEntropyLoss
    optimizer = SGD(model.parameters(), lr=0.01)

    # 训练模型
    train_model(model, train_loader, optimizer, loss_fn, epochs=10)

    # 测试模型
    test_model(model, test_loader)
