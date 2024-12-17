# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

import os
import requests
import numpy as np
import pandas as pd

import orion.nn as nn
from orion.data import DataLoader
from orion.nn.loss import CrossEntropyLoss
from orion.nn.optim import GD


def download_mnist_csv(save_dir="../datasets"):
    """
    Download the MNIST dataset in CSV format.

    :param save_dir: Directory where the dataset will be saved.
    :return: Paths to the downloaded training and test CSV files.
    """
    # URLs for the dataset
    train_url = "http://www.pjreddie.com/media/files/mnist_train.csv"
    test_url = "http://www.pjreddie.com/media/files/mnist_test.csv"

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "mnist_train.csv")
    test_path = os.path.join(save_dir, "mnist_test.csv")

    # Download training set
    if not os.path.exists(train_path):
        print("Downloading MNIST training set...")
        with open(train_path, "wb") as f:
            response = requests.get(train_url, stream=True)
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
        print(f"Training set downloaded: {train_path}")
    else:
        print(f"Training set already exists: {train_path}")

    # Download test set
    if not os.path.exists(test_path):
        print("Downloading MNIST test set...")
        with open(test_path, "wb") as f:
            response = requests.get(test_url, stream=True)
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
        print(f"Test set downloaded: {test_path}")
    else:
        print(f"Test set already exists: {test_path}")

    return train_path, test_path


def load_mnist_csv(train_path, test_path):
    """
    Load the MNIST dataset from CSV files.

    :param train_path: Path to the training CSV file.
    :param test_path: Path to the test CSV file.
    :return: A tuple containing training and test data (features and labels).
    """
    # Ensure the dataset is downloaded
    download_mnist_csv()

    # Load training data
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values / 255.0  # Normalize pixel values
    y_train = train_data.iloc[:, 0].values          # Labels

    # Load test data
    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values

    return (X_train, y_train), (X_test, y_test)


class Model(nn.Module):
    """
    A simple neural network model with two linear layers and a ReLU activation.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, initializer='he', label='fc1')
        self.relu1 = nn.ReLU(label='relu1')
        self.fc2 = nn.Linear(128, 10, initializer='he', label='fc2')

    def forward(self, x):
        """
        Forward pass through the model.

        :param x: Input tensor.
        :return: Output tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def train_model(model, dataloader, optimizer, loss_fn, epochs=10):
    """
    Train the model for a specified number of epochs.

    :param model: The neural network model to train.
    :param dataloader: DataLoader for training data.
    :param optimizer: Optimizer for parameter updates.
    :param loss_fn: Loss function for training.
    :param epochs: Number of training epochs.
    """
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.data

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def test_model(model, test_loader):
    """
    Evaluate the model on the test dataset.

    :param model: The trained neural network model.
    :param test_loader: DataLoader for test data.
    """
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        predictions = np.argmax(y_pred.data, axis=1)
        correct += np.sum(predictions == y_batch.data)
        total += y_batch.shape[0]

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # Paths to dataset files
    train_path = "../datasets/mnist_train.csv"
    test_path = "../datasets/mnist_test.csv"

    # Load the dataset
    (X_train, y_train), (X_test, y_test) = load_mnist_csv(train_path, test_path)

    # Create DataLoader instances
    batch_size = 32
    train_loader = DataLoader((X_train, y_train), batch_size=batch_size)
    test_loader = DataLoader((X_test, y_test), batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    model = Model()
    loss_fn = CrossEntropyLoss
    optimizer = GD(model.parameters(), lr=0.5)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, optimizer, loss_fn, epochs=20)

    # Evaluate the model
    print("Testing the model...")
    test_model(model, test_loader)
