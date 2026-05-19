# API Reference

## nn.core

### `Tensor`

`nn/core/tensor.py`

Base data container wrapping a NumPy `float64` array.

```python
Tensor(data: np.ndarray, requiresGrad: bool = False)
```

| Member | Type | Description |
| ------ | ---- | ----------- |
| `.Data` | `np.ndarray` | The underlying float64 array |
| `.Grad` | `np.ndarray \| None` | Gradient array (same shape as `.Data`); `None` if `RequiresGrad=False` |
| `.RequiresGrad` | `bool` | Whether this tensor participates in gradient computation |
| `.Shape` | `tuple` | Shape of `.Data` |

---

### `Parameter`

`nn/core/parameter.py`

A `Tensor` subclass representing a trainable parameter. Always has `RequiresGrad=True`.

```python
Parameter(data: np.ndarray)
```

Inherits all `Tensor` members. Used for layer weights and biases.

---

### `Module`

`nn/core/module.py`

Abstract base class for all layers and containers.

```python
class Module(ABC)
```

| Method / Property | Signature | Description |
| ----------------- | --------- | ----------- |
| `Forward` | `(x: Tensor) -> Tensor` | **Abstract.** Computes the forward pass. |
| `Backward` | `(dZ: np.ndarray) -> np.ndarray` | **Abstract.** Computes gradients and returns upstream gradient. |
| `Parameters` | `-> list[Parameter]` | Returns all parameters recursively (including child modules). |
| `AddParameter` | `(name: str, param: Parameter) -> None` | Registers a parameter on this module. |
| `AddModule` | `(name: str, module: Module) -> None` | Registers a child module. |

---

## nn.layers

### `Linear`

`nn/layers/linear.py`

Fully-connected layer: $Z = WX + B$.

```python
Linear(inputDim: int, outputDim: int)
```

| Member | Description |
| ------ | ----------- |
| `W` | `Parameter` of shape `(outputDim, inputDim)` — He-initialized |
| `B` | `Parameter` of shape `(outputDim, 1)` — zero-initialized |

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `Forward` | `(x: Tensor) -> Tensor` | Returns `Z = W @ x.Data + B.Data` as a `Tensor`. Stores input. |
| `Backward` | `(dZ: np.ndarray) -> np.ndarray` | Accumulates `W.Grad`, `B.Grad`; returns `dX = Wᵀ @ dZ`. |

Raises `ValueError` if `x.Shape[0] != inputDim`.  
Raises `RuntimeError` if `Backward` is called before `Forward`.

---

### `Sequential`

`nn/layers/sequential.py`

Chains modules in order.

```python
Sequential(*modules: Module)
```

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `Forward` | `(x: Tensor) -> Tensor` | Passes `x` through each module in order. |
| `Backward` | `(dZ: np.ndarray) -> np.ndarray` | Passes gradient through modules in reverse order. |
| `Parameters` | `-> list[Parameter]` | Collects parameters from all child modules. |

---

## nn.activations

All activations share the same interface:

```python
activation = ReLU()          # stateless constructor
out = activation.Forward(x)  # x: Tensor → Tensor
grad = activation.Backward(dZ)  # dZ: np.ndarray → np.ndarray
```

Raises `RuntimeError` if `Backward` is called before `Forward`.

### `Sigmoid`

`nn/activations/sigmoid.py`

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Backward: $dZ \cdot a(1-a)$ where $a$ is the stored forward output.

### `Tanh`

`nn/activations/tanh.py`

$$\tanh(x)$$

Backward: $dZ \cdot (1 - a^2)$ where $a$ is the stored forward output.

### `ReLU`

`nn/activations/relu.py`

$$\text{ReLU}(x) = \max(0, x)$$

Backward: $dZ \cdot \mathbf{1}[x > 0]$ where the mask is derived from the stored forward input.

### `Softmax`

`nn/activations/softmax.py`

$$\text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

Numerically stable (max-shifted). Operates column-wise for batched input.

Backward: $a \odot (dZ - \sum(dZ \odot a))$ where $a$ is the stored forward output.

---

## nn.losses

### `MSELoss`

`nn/losses/mse.py`

Mean Squared Error loss.

```python
loss_fn = MSELoss()
loss = loss_fn(y_pred, y_true)   # equivalent to loss_fn.Forward(y_pred, y_true)
dZ   = loss_fn.Backward()
```

| Method | Signature | Description |
| ------ | --------- | ----------- |
| `Forward` | `(yPred: Tensor, yTrue: Tensor) -> Tensor` | Returns scalar loss `Tensor`. |
| `Backward` | `() -> np.ndarray` | Returns $\frac{2}{n}(\hat{y} - y)$ — the gradient w.r.t. predictions. |
| `__call__` | `(yPred, yTrue) -> Tensor` | Delegates to `Forward`. |

Raises `RuntimeError` if `Backward` is called before `Forward`.

---

## Gradient Checking

Tests verify gradients numerically using the finite-difference approximation:

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}$$

All analytical gradients are verified to match numerical gradients within `1e-6` absolute tolerance. See `tests/` for examples.
