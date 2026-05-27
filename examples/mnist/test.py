"""Evaluate a saved MNIST model on the raw test set.

Knows nothing about the model architecture — relies on `model.InputShape`
to project raw `(N, 1, 28, 28)` images into whatever layout the model wants.

Run:
    python examples/mnist/test.py                      # default: mnist_model.npz
    python examples/mnist/test.py --model some.npz
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nn import Load, Tensor


HERE = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.join(HERE, "mnist_model.npz")
TEST_CSV = os.path.join(HERE, "csv", "test.csv")


def load_test_csv(path):
    print(f"Loading {os.path.basename(path)}...", flush=True)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    labels = data[:, 0].astype(np.int32)
    pixels = (data[:, 1:] / 255.0).astype(np.float32)
    return pixels.reshape(-1, 1, 28, 28), labels


def to_model_input(images_nchw: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Reshape (N, 1, 28, 28) into the model's declared `input_shape`.

    Batch axis in `input_shape` is marked with None.
    """
    batch_axis = input_shape.index(None)
    per_sample = tuple(s for s in input_shape if s is not None)
    N = images_nchw.shape[0]
    reshaped = images_nchw.reshape((N,) + per_sample)
    return np.moveaxis(reshaped, 0, batch_axis).astype(np.float32)


def evaluate(model, images_nchw, labels, batch=512):
    shape = model.InputShape
    if shape is None:
        sys.exit(
            "Model has no InputShape declared — cannot auto-feed. "
            "Re-train with an updated build_*() that passes inputShape=..."
        )
    N = images_nchw.shape[0]
    correct = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    for start in range(0, N, batch):
        chunk = images_nchw[start:start + batch]
        truth = labels[start:start + batch]
        preds = np.argmax(model(Tensor(to_model_input(chunk, shape))).Data, axis=0)
        correct += int((preds == truth).sum())
        for t, p in zip(truth, preds):
            confusion[t, p] += 1
    return correct / N * 100.0, confusion


def print_confusion(cm: np.ndarray) -> None:
    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "      " + "  ".join(f"{c:>4d}" for c in range(10))
    print(header)
    print("     " + "-" * (len(header) - 5))
    for r in range(10):
        row = "  ".join(f"{cm[r, c]:>4d}" for c in range(10))
        print(f"  {r} | {row}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Path to saved .npz model (default: {os.path.basename(DEFAULT_MODEL)}).")
    parser.add_argument("--no-confusion", action="store_true",
                        help="Skip printing the per-class confusion matrix.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"No model at {args.model}.")

    print(f"Loading model from {os.path.basename(args.model)}...")
    model = Load(args.model)
    print(f"Model expects input shape: {model.InputShape}")

    test_images, test_labels = load_test_csv(TEST_CSV)
    acc, cm = evaluate(model, test_images, test_labels)
    print(f"\nTest accuracy on raw data: {acc:.2f}%  ({int(cm.sum().item())} samples)")

    if not args.no_confusion:
        print_confusion(cm)


if __name__ == "__main__":
    main()
