"""MNIST digit classification — fully autograd-driven."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nn import (
    Sequential, Linear, ReLU, Softmax, CELoss, Tensor,
    TensorDataset, DataLoader,
)
from nn.optim import Adam
from nn.init import HeInitialization


def load_csv(path):
    print(f"Loading {os.path.basename(path)}...", flush=True)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    labels = data[:, 0].astype(np.int32)
    pixels = data[:, 1:] / 255.0
    return pixels, labels


def one_hot(labels, num_classes=10):
    oh = np.zeros((num_classes, len(labels)), dtype=np.float32)
    oh[labels, np.arange(len(labels))] = 1.0
    return oh


def accuracy(model, X, labels):
    preds = np.argmax(model(X).Data, axis=0)
    return (preds == labels).mean() * 100.0


base = os.path.dirname(__file__)
train_pixels, train_labels = load_csv(os.path.join(base, "csv", "train.csv"))
test_pixels,  test_labels  = load_csv(os.path.join(base, "csv", "test.csv"))

X_train = Tensor(train_pixels.T)
y_train = Tensor(one_hot(train_labels))
X_test  = Tensor(test_pixels.T)

loader = DataLoader(TensorDataset(X_train, y_train), batchSize=128, shuffle=True)

model = Sequential(
    Linear(784, 256, HeInitialization),
    ReLU(),
    Linear(256, 128, HeInitialization),
    ReLU(),
    Linear(128, 10, HeInitialization),
    Softmax(),
)

criterion = CELoss()
optimizer = Adam(model.Parameters, lr=1e-3)

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for X_batch, y_batch in loader:
        out = model(X_batch)
        loss = criterion(out, y_batch)
        total_loss += float(loss.Data)

        optimizer.ZeroGrad()
        loss.backward()
        optimizer.Step()

    train_acc = accuracy(model, X_train, train_labels)
    print(f"epoch [{epoch:2d}/{EPOCHS}]  loss: {total_loss / len(loader):.4f}  train_acc: {train_acc:.1f}%")

test_acc = accuracy(model, X_test, test_labels)
print(f"\nTest accuracy: {test_acc:.1f}%")
