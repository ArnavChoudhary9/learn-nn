"""MNIST digit classification — fully autograd-driven, with save/load."""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nn import (
    Sequential, Linear, ReLU, Softmax, CELoss, Tensor,
    TensorDataset, DataLoader, Save, Load,
)
from nn.optim import Adam
from nn.init import HeInitialization


HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "mnist_model.npz")


def load_csv(path):
    print(f"Loading {os.path.basename(path)}...", flush=True)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    labels = data[:, 0].astype(np.int32)
    pixels = data[:, 1:] / 255.0
    return pixels, labels


def preprocess(pixels, mean=None):
    """Zero-center pixels using the training-set mean.

    After centering, the (zero) background remains at value 0, so jitter
    padding with zeros stays visually consistent with the original image.
    """
    if mean is None:
        mean = float(pixels.mean())
    return (pixels - mean).astype(np.float32), mean


def augment(pixels, max_shift=2, noise_std=0.05):
    """Random per-image jitter + Gaussian noise. Input/output shape: (N, 784)."""
    N = pixels.shape[0]
    imgs = pixels.reshape(N, 28, 28)
    padded = np.pad(imgs, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    dy = np.random.randint(0, 2 * max_shift + 1, size=N)
    dx = np.random.randint(0, 2 * max_shift + 1, size=N)
    out = np.empty_like(imgs)
    for i in range(N):
        out[i] = padded[i, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28]
    out += np.random.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    return out.reshape(N, 784)


def one_hot(labels, num_classes=10):
    oh = np.zeros((num_classes, len(labels)), dtype=np.float32)
    oh[labels, np.arange(len(labels))] = 1.0
    return oh


def accuracy(model, X, labels):
    preds = np.argmax(model(X).Data, axis=0)
    return (preds == labels).mean() * 100.0


def build_model():
    return Sequential(
        Linear(784, 256, HeInitialization),
        ReLU(),
        Linear(256, 128, HeInitialization),
        ReLU(),
        Linear(128, 10, HeInitialization),
        Softmax(),
    )


def train(model, train_pixels, y_train, train_labels, epochs):
    X_train_clean = Tensor(train_pixels.T)
    criterion = CELoss()
    optimizer = Adam(model.Parameters, lr=1e-3)

    for epoch in range(1, epochs + 1):
        aug_pixels = augment(train_pixels)
        X_train_aug = Tensor(aug_pixels.T)
        loader = DataLoader(TensorDataset(X_train_aug, y_train), batchSize=128, shuffle=True)

        total_loss = 0.0
        for X_batch, y_batch in loader:
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += float(loss.Data)

            optimizer.ZeroGrad()
            loss.backward()
            optimizer.Step()

        train_acc = accuracy(model, X_train_clean, train_labels)
        print(f"epoch [{epoch:2d}/{epochs}]  loss: {total_loss / len(loader):.4f}  "
              f"train_acc: {train_acc:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
                        help="Skip training; load weights from disk and just evaluate.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving the model after training.")
    args = parser.parse_args()

    train_pixels, train_labels = load_csv(os.path.join(HERE, "csv", "train.csv"))
    test_pixels,  test_labels  = load_csv(os.path.join(HERE, "csv", "test.csv"))

    train_pixels, train_mean = preprocess(train_pixels)
    test_pixels,  _          = preprocess(test_pixels, mean=train_mean)

    y_train = Tensor(one_hot(train_labels))
    X_test  = Tensor(test_pixels.T)

    if args.load:
        if not os.path.exists(MODEL_PATH):
            sys.exit(f"No saved model at {MODEL_PATH} — run without --load first.")
        print(f"Loading model from {os.path.basename(MODEL_PATH)}...")
        model = Load(MODEL_PATH)
    else:
        model = build_model()
        train(model, train_pixels, y_train, train_labels, args.epochs)
        if not args.no_save:
            Save(model, MODEL_PATH)
            size_kb = os.path.getsize(MODEL_PATH) / 1024
            print(f"\nSaved model to {os.path.basename(MODEL_PATH)} ({size_kb:.1f} KB)")

    test_acc = accuracy(model, X_test, test_labels)
    print(f"\nTest accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
