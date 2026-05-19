# neural_net

A from-scratch neural network library built on NumPy. No autograd magic — just clean manual implementations of forward passes, backpropagation, and gradient descent, built for learning how neural networks actually work.

## Features

- **Tensors & Parameters** — NumPy-backed tensor with gradient tracking
- **Layers** — `Linear` (fully-connected), `Sequential` container
- **Activations** — `Sigmoid`, `Tanh`, `ReLU`, `Softmax`
- **Losses** — `MSELoss`
- **Modular design** — `Module` base class mirroring PyTorch's API

## Installation

```bash
pip install -e .
```

Requires Python 3.9+ and NumPy.

## Quick Start

```python
import numpy as np
from nn.core.tensor import Tensor
from nn.layers.linear import Linear
from nn.layers.sequential import Sequential
from nn.activations.relu import ReLU
from nn.losses.mse import MSELoss

# Build a 2-layer MLP
model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
)

# Forward pass
x = Tensor(np.array([[1.0, 2.0]]).T)  # shape (2, 1)
y = Tensor(np.array([[1.0]]))

loss_fn = MSELoss()
pred = model.Forward(x)
loss = loss_fn(pred, y)
print(loss.Data)  # scalar loss value

# Backward pass
dZ = loss_fn.Backward()
model.Backward(dZ)

# Gradient descent step
lr = 0.01
for p in model.Parameters():
    p.Data -= lr * p.Grad
```

## Project Structure

```text
nn/
├── core/
│   ├── tensor.py         # Tensor — NumPy ndarray wrapper with gradient tracking
│   ├── parameter.py      # Parameter — trainable Tensor subclass
│   └── module.py         # Module — abstract base class for all layers
├── activations/
│   ├── relu.py           # ReLU: max(0, x)
│   ├── sigmoid.py        # Sigmoid: 1 / (1 + exp(-x))
│   ├── tanh.py           # Tanh: tanh(x)
│   └── softmax.py        # Softmax (numerically stable)
├── layers/
│   ├── linear.py         # Fully-connected layer: Z = WX + B
│   └── sequential.py     # Sequential container
└── losses/
    └── mse.py            # Mean Squared Error loss

tests/
├── test_core.py          # Tensor, Parameter, Module tests
├── test_activations.py   # Activation forward + backward tests
├── test_layers.py        # Linear, Sequential tests + gradient checks
└── test_loss.py          # MSELoss tests + gradient checks

docs/
├── math/
│   └── 001_backpropagation_foundations.md   # Math: forward prop, chain rule, backprop
└── implementation/
    ├── architecture.md                       # Design overview and module structure
    └── api_reference.md                      # Class and method reference
```

## Running Tests

```bash
pytest tests/
```

All tests use numerical gradient checking (`< 1e-6` tolerance) to verify backpropagation correctness.

## Documentation

| Doc | Description |
| --- | ----------- |
| [Backpropagation Foundations](docs/math/001_backpropagation_foundations.md) | Math derivations: perceptron, chain rule, forward/backward propagation |
| [Architecture](docs/implementation/architecture.md) | Design decisions, module system, data flow |
| [API Reference](docs/implementation/api_reference.md) | All classes, constructors, and method signatures |

## Roadmap

Planned but not yet implemented:

| Module | Path |
| ------ | ---- |
| SGD, Momentum, Adam optimizers | `nn/optim/` |
| Xavier / He weight initializers | `nn/init/` |
| Autograd engine | `nn/autograd/` |
| Cross-Entropy, Binary Cross-Entropy | `nn/losses/` |
| Dataset, DataLoader | `nn/data/` |
| Trainer, Callbacks, Metrics | `nn/training/` |
| Flatten layer | `nn/layers/` |
