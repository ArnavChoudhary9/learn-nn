"""MNIST digit classification — MLP by default, CNN with --cnn.

The training set is augmented (random rotation, jitter, per-pixel noise);
the test set is fed to the model raw, with no transformation at all.

The saved model carries its expected `InputShape`, so callers don't need to
know whether they're using the MLP or CNN to feed it.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nn import (
    Sequential, Linear, Conv2D, MaxPool2D, Flatten, ReLU, Softmax, CELoss, Tensor,
    TensorDataset, DataLoader, Save, Load,
)
from nn.optim import Adam
from nn.init import HeInitialization


HERE = os.path.dirname(__file__)
MLP_MODEL_PATH = os.path.join(HERE, "mnist_model.npz")
CNN_MODEL_PATH = os.path.join(HERE, "mnist_cnn_model.npz")


def load_csv(path):
    print(f"Loading {os.path.basename(path)}...", flush=True)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    labels = data[:, 0].astype(np.int32)
    pixels = data[:, 1:] / 255.0
    return pixels.astype(np.float32), labels


# ---------------------------------------------------------------------------
# Training-only augmentation
# ---------------------------------------------------------------------------


def _RotateBatch(images: np.ndarray, max_deg: float) -> np.ndarray:
    """Rotate each (1, 28, 28) image by an independent random angle.

    Nearest-neighbor inverse-warp around the image center. Pure numpy.
    """
    N, _, H, W = images.shape
    angles = np.random.uniform(-max_deg, max_deg, size=N).astype(np.float32)
    cos = np.cos(np.deg2rad(angles))
    sin = np.sin(np.deg2rad(angles))

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dy = (y - cy).astype(np.float32)
    dx = (x - cx).astype(np.float32)

    # Inverse mapping: for each output pixel find the source pixel.
    src_y = cos[:, None, None] * dy[None] + sin[:, None, None] * dx[None] + cy
    src_x = -sin[:, None, None] * dy[None] + cos[:, None, None] * dx[None] + cx
    src_y = np.clip(np.round(src_y).astype(np.int32), 0, H - 1)
    src_x = np.clip(np.round(src_x).astype(np.int32), 0, W - 1)

    n_idx = np.arange(N)[:, None, None]
    return images[n_idx, 0, src_y, src_x][:, None, :, :]


def augment(images_nchw, max_shift=2, max_rot_deg=12.0, noise_std=0.05, salt_prob=0.02):
    """Random rotation + translation + Gaussian noise + sparse salt-pepper.

    Input/output shape: (N, 1, 28, 28). Training only — never call on test data.
    """
    out = _RotateBatch(images_nchw, max_rot_deg)

    N = out.shape[0]
    padded = np.pad(out, ((0, 0), (0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    dy = np.random.randint(0, 2 * max_shift + 1, size=N)
    dx = np.random.randint(0, 2 * max_shift + 1, size=N)
    shifted = np.empty_like(out)
    for i in range(N):
        shifted[i] = padded[i, :, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28]

    shifted += np.random.normal(0.0, noise_std, size=shifted.shape).astype(np.float32)

    # Salt-and-pepper on a random subset of pixels.
    mask = np.random.rand(*shifted.shape) < salt_prob
    salt = np.random.rand(*shifted.shape).astype(np.float32)
    shifted = np.where(mask, salt, shifted)

    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Layout glue — projects an (N, 1, 28, 28) batch onto whatever the model wants
# ---------------------------------------------------------------------------


def to_model_input(pixels_nchw: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Reshape (N, 1, 28, 28) into the model's declared `input_shape`.

    The batch axis in `input_shape` is marked with None.
    """
    batch_axis = input_shape.index(None)
    per_sample_shape = tuple(s for s in input_shape if s is not None)
    N = pixels_nchw.shape[0]
    flat_per_sample = int(np.prod(per_sample_shape))
    reshaped = pixels_nchw.reshape(N, flat_per_sample).reshape((N,) + per_sample_shape)
    # Move the batch axis from position 0 to where the model expects it.
    return np.moveaxis(reshaped, 0, batch_axis).astype(np.float32)


def one_hot(labels, num_classes=10):
    oh = np.zeros((num_classes, len(labels)), dtype=np.float32)
    oh[labels, np.arange(len(labels))] = 1.0
    return oh


def accuracy(model, images_nchw, labels, batch=512):
    """Batched eval — full-set forward through Conv2D's im2col buffer would OOM."""
    shape = model.InputShape
    N = images_nchw.shape[0]
    correct = 0
    for start in range(0, N, batch):
        chunk = images_nchw[start:start + batch]
        x = Tensor(to_model_input(chunk, shape))
        preds = np.argmax(model(x).Data, axis=0)
        correct += int((preds == labels[start:start + batch]).sum())
    return correct / N * 100.0


# ---------------------------------------------------------------------------
# Models — each declares its external input contract via `inputShape`
# ---------------------------------------------------------------------------


def build_mlp():
    return Sequential(
        Linear(784, 256, HeInitialization),
        ReLU(),
        Linear(256, 128, HeInitialization),
        ReLU(),
        Linear(128, 10, HeInitialization),
        Softmax(),
        inputShape=(784, None),
    )


def build_cnn():
    return Sequential(
        Conv2D(1, 8, kernelSize=3, stride=1),
        ReLU(),
        MaxPool2D(2),
        Conv2D(8, 16, kernelSize=3, stride=1),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(16 * 5 * 5, 64, HeInitialization),
        ReLU(),
        Linear(64, 10, HeInitialization),
        Softmax(),
        inputShape=(None, 1, 28, 28),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(model, train_images_nchw, y_train, train_labels, epochs, batch_size):
    criterion = CELoss()
    optimizer = Adam(model.Parameters, lr=1e-3)
    shape = model.InputShape

    x_batch_axis = shape.index(None)

    for epoch in range(1, epochs + 1):
        aug = augment(train_images_nchw)
        X_aug = Tensor(to_model_input(aug, shape))
        dataset = TensorDataset(X_aug, y_train, batchAxis=(x_batch_axis, -1))
        loader = DataLoader(dataset, batchSize=batch_size, shuffle=True)

        total_loss = 0.0
        for X_batch, y_batch in loader:
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += float(loss.Data)

            optimizer.ZeroGrad()
            loss.backward()
            optimizer.Step()

        train_acc = accuracy(model, train_images_nchw, train_labels)
        print(f"epoch [{epoch:2d}/{epochs}]  loss: {total_loss / len(loader):.4f}  "
              f"train_acc: {train_acc:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", action="store_true",
                        help="Train a small CNN instead of the default MLP.")
    parser.add_argument("--load", action="store_true",
                        help="Skip training; load weights from disk and just evaluate.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Defaults to 10 (MLP) or 5 (CNN).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving the model after training.")
    args = parser.parse_args()

    use_cnn = args.cnn
    epochs = args.epochs if args.epochs is not None else (5 if use_cnn else 10)
    model_path = CNN_MODEL_PATH if use_cnn else MLP_MODEL_PATH

    train_pixels, train_labels = load_csv(os.path.join(HERE, "csv", "train.csv"))
    test_pixels,  test_labels  = load_csv(os.path.join(HERE, "csv", "test.csv"))

    # Canonical layout is NCHW with raw [0,1] pixels.
    # No mean-centering: the test set is fed to the model exactly as loaded.
    train_images = train_pixels.reshape(-1, 1, 28, 28)
    test_images  = test_pixels.reshape(-1, 1, 28, 28)

    y_train = Tensor(one_hot(train_labels))

    if args.load:
        if not os.path.exists(model_path):
            sys.exit(f"No saved model at {model_path} — run without --load first.")
        print(f"Loading model from {os.path.basename(model_path)}...")
        model = Load(model_path)
    else:
        model = build_cnn() if use_cnn else build_mlp()
        train(model, train_images, y_train, train_labels, epochs, args.batch_size)
        if not args.no_save:
            Save(model, model_path)
            size_kb = os.path.getsize(model_path) / 1024
            print(f"\nSaved model to {os.path.basename(model_path)} ({size_kb:.1f} KB)")

    test_acc = accuracy(model, test_images, test_labels)
    print(f"\nTest accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
