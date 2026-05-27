# MNIST

End-to-end MNIST digit classification using this repo's pure-NumPy autograd
engine. Trains either a small MLP or a small CNN, saves the model to `.npz`,
and lets you evaluate it on the test set or draw digits with the mouse.

## 1. Get the data

The scripts expect two CSV files under `examples/mnist/csv/`:

```
examples/mnist/csv/train.csv     # 60 000 rows
examples/mnist/csv/test.csv      # 10 000 rows
```

Each row is `label, pixel0, pixel1, ..., pixel783` with pixel values in `0–255`
and a header line.

### Option A — Kaggle "Digit Recognizer" CSVs (easiest)

The Kaggle competition [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)
publishes MNIST as CSVs in exactly the layout above. The `train.csv` it ships
is the full 60 000-row training set; the `test.csv` it ships is unlabeled, so
use it only for the competition — not for the `test.py` script in this folder
(which needs labels).

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/  (Windows: %USERPROFILE%\.kaggle\)
kaggle competitions download -c digit-recognizer -p examples/mnist/csv
cd examples/mnist/csv
unzip digit-recognizer.zip
```

For a labeled test set, download the standard MNIST CSV from Kaggle's
["MNIST in CSV"](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
dataset instead — it ships both `mnist_train.csv` and `mnist_test.csv`
already in the right shape. Rename them to `train.csv` and `test.csv`.

### Option B — original IDX files → CSV

If you grabbed MNIST as the original IDX binaries (e.g. from
`yann.lecun.com/exdb/mnist` mirrors or Kaggle's
["MNIST Dataset"](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)),
drop the four files in `examples/mnist/`:

```
examples/mnist/train-images.idx3-ubyte
examples/mnist/train-labels.idx1-ubyte
examples/mnist/t10k-images.idx3-ubyte
examples/mnist/t10k-labels.idx1-ubyte
```

Then convert:

```bash
python examples/mnist/convert_to_csv.py
```

This writes `csv/train.csv` (60 000 rows) and `csv/test.csv` (10 000 rows).

## 2. Train

```bash
# MLP — default, ~10 epochs
python examples/mnist/train.py

# CNN — small Conv→Pool→Conv→Pool→FC, ~5 epochs by default
python examples/mnist/train.py --cnn
```

Training augments each epoch with random rotation, translation, gaussian noise,
and sparse salt-and-pepper noise. The test set is fed to the model raw.

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--cnn` | off | train a CNN instead of the MLP |
| `--epochs N` | 10 (MLP) / 5 (CNN) | epoch count |
| `--batch-size N` | 128 | mini-batch size |
| `--no-save` | off | skip writing the `.npz` after training |
| `--load` | off | skip training; just load the saved model and report test accuracy |

Models are written to:

```
examples/mnist/mnist_model.npz        # MLP
examples/mnist/mnist_cnn_model.npz    # CNN
```

The saved file embeds the model's expected `InputShape`, so downstream
scripts don't need to know whether it's an MLP or a CNN.

## 3. Evaluate

`test.py` runs a saved model on `csv/test.csv` and prints accuracy plus a
10×10 confusion matrix.

```bash
# Default: examples/mnist/mnist_model.npz (the MLP)
python examples/mnist/test.py

# Point at any other saved model
python examples/mnist/test.py --model examples/mnist/mnist_cnn_model.npz

# Skip the confusion matrix
python examples/mnist/test.py --no-confusion
```

## 4. Visualize (draw digits with the mouse)

`visualize.py` opens a Tk window with a drawing canvas on the left and live
per-class probability bars on the right. Stroke with the left mouse button;
"Clear" wipes the canvas.

```bash
# MLP model (default)
python examples/mnist/visualize.py

# CNN model
python examples/mnist/visualize.py --cnn

# Any saved .npz
python examples/mnist/visualize.py --model path/to/model.npz
```

The visualizer reads `model.InputShape` to figure out how to feed the
drawing, so the same script works for both architectures with no flags
beyond which file to load.

## Layout

```
examples/mnist/
├── csv/                       # train.csv + test.csv go here
├── convert_to_csv.py          # IDX → CSV converter
├── train.py                   # trains MLP (default) or CNN (--cnn)
├── test.py                    # evaluates a saved model on csv/test.csv
├── visualize.py               # interactive draw-and-predict window
├── mnist_model.npz            # written by `train.py`
└── mnist_cnn_model.npz        # written by `train.py --cnn`
```
