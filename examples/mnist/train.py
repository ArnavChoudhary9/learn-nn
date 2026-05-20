"""MNIST digit classification."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nn import Sequential, Linear, ReLU, Softmax, CELoss, SGD, Tensor, TensorDataset, DataLoader
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
    preds = np.argmax(model.Forward(X).Data, axis=0)
    return (preds == labels).mean() * 100.0


base = os.path.dirname(__file__)
train_pixels, train_labels = load_csv(os.path.join(base, "csv", "train.csv"))
test_pixels,  test_labels  = load_csv(os.path.join(base, "csv", "test.csv"))

X_train = Tensor(train_pixels.T)         # (784, 60000)
y_train = Tensor(one_hot(train_labels))  # (10,  60000)
X_test  = Tensor(test_pixels.T)          # (784, 10000)

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
optimizer = SGD(model.Parameters, lr=0.1)

epochs = 50
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for X_batch, y_batch in loader:
        outputs = model.Forward(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += float(loss.Data)

        optimizer.ZeroGrad()
        model.Backward(criterion.Backward())
        optimizer.Step()

    train_acc = accuracy(model, X_train, train_labels)
    print(f"Epoch [{epoch:2d}/{epochs}]  loss: {total_loss / len(loader):.4f}  train_acc: {train_acc:.1f}%")

test_acc = accuracy(model, X_test, test_labels)
print(f"\nTest accuracy: {test_acc:.1f}%")
