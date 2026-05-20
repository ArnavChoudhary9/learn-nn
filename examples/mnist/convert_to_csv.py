"""Convert MNIST IDX binary files to CSV format."""
import struct
import numpy as np
import csv
import os


def read_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803, f"Bad magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def read_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801, f"Bad magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def write_csv(images, labels, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"pixel{i}" for i in range(images.shape[1])]
        writer.writerow(header)
        for label, pixels in zip(labels, images):
            writer.writerow([int(label)] + pixels.tolist())
    print(f"Wrote {len(labels)} rows to {out_path}")


base = os.path.dirname(__file__)

train_images = read_images(os.path.join(base, "train-images.idx3-ubyte"))
train_labels = read_labels(os.path.join(base, "train-labels.idx1-ubyte"))
write_csv(train_images, train_labels, os.path.join(base, "csv", "train.csv"))

test_images = read_images(os.path.join(base, "t10k-images.idx3-ubyte"))
test_labels = read_labels(os.path.join(base, "t10k-labels.idx1-ubyte"))
write_csv(test_images, test_labels, os.path.join(base, "csv", "test.csv"))
