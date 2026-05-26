"""Autograd engine demo — three progressively complex examples."""

import numpy as np
from nn.core.tensor import Tensor


# ---------------------------------------------------------------------------
# Example 1: scalar arithmetic — verify chain rule by hand
# ---------------------------------------------------------------------------

print("=" * 55)
print("Example 1: scalar chain rule")
print("=" * 55)

x = Tensor([3.0], requiresGrad=True)
y = Tensor([4.0], requiresGrad=True)

# z = (x * y) + (x ** 2)   =>  dz/dx = y + 2x = 4 + 6 = 10
#                               dz/dy = x = 3
z = (x * y) + (x ** 2)
z.backward()

print(f"x={x.Data[0]}, y={y.Data[0]}")
print(f"z = x*y + x^2 = {z.Data[0]}")
print(f"dz/dx (expected 10): {x.Grad[0]}")
print(f"dz/dy (expected  3): {y.Grad[0]}")


# ---------------------------------------------------------------------------
# Example 2: MSE loss on a single linear layer, manual gradient check
# ---------------------------------------------------------------------------

print()
print("=" * 55)
print("Example 2: linear layer + MSE, gradient check")
print("=" * 55)

np.random.seed(0)
W = Tensor(np.random.randn(2, 3).astype(np.float32), requiresGrad=True)
b = Tensor(np.zeros((2, 1), dtype=np.float32), requiresGrad=True)
X = Tensor(np.random.randn(3, 4).astype(np.float32))          # (in, batch)
Y = Tensor(np.random.randn(2, 4).astype(np.float32))          # (out, batch)

pred = (W @ X) + b                     # (2, 4)
diff = pred - Y
loss = (diff ** 2).mean()
loss.backward()

print(f"loss : {loss.Data:.4f}")
print(f"W.Grad shape : {W.Grad.shape}")   # (2, 3)
print(f"b.Grad shape : {b.Grad.shape}")   # (2, 1)

# numerical gradient check for W[0,0]
eps = 1e-4
W_plus = Tensor(W.Data.copy(), requiresGrad=False)
W_plus.Data[0, 0] += eps
loss_plus = (((W_plus @ X) + b) - Y) ** 2
lp = float(np.mean(loss_plus.Data))

W_minus = Tensor(W.Data.copy(), requiresGrad=False)
W_minus.Data[0, 0] -= eps
loss_minus = (((W_minus @ X) + b) - Y) ** 2
lm = float(np.mean(loss_minus.Data))

numerical_grad = (lp - lm) / (2 * eps)
analytical_grad = float(W.Grad[0, 0])
print(f"W[0,0] numerical grad : {numerical_grad:.6f}")
print(f"W[0,0] analytical grad: {analytical_grad:.6f}")
rel_err = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8)
print(f"relative error: {rel_err:.2e}  {'OK' if rel_err < 5e-3 else 'FAIL'}")  # float32 tolerance


# ---------------------------------------------------------------------------
# Example 3: training a 1-layer network on XOR with SGD
# ---------------------------------------------------------------------------

print()
print("=" * 55)
print("Example 3: XOR with 1 hidden layer + autograd SGD")
print("=" * 55)

np.random.seed(42)

# XOR data  — inputs (2, 4), targets (1, 4)
X_xor = Tensor(np.array([[0,0,1,1],[0,1,0,1]], dtype=np.float32))
Y_xor = Tensor(np.array([[0,1,1,0]], dtype=np.float32))

# Parameters
W1 = Tensor(np.random.randn(4, 2).astype(np.float32) * 0.5, requiresGrad=True)
b1 = Tensor(np.zeros((4, 1), dtype=np.float32), requiresGrad=True)
W2 = Tensor(np.random.randn(1, 4).astype(np.float32) * 0.5, requiresGrad=True)
b2 = Tensor(np.zeros((1, 1), dtype=np.float32), requiresGrad=True)

params = [W1, b1, W2, b2]
lr = 1.0


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


for step in range(1, 5001):
    # Zero grads
    for p in params:
        p.Grad = np.zeros(p.Shape, dtype=np.float32)

    # Forward  (using autograd ops via operator overloads)
    from nn.autograd.ops.sigmoid import Sigmoid
    h = Sigmoid.apply(W1 @ X_xor + b1)      # (4, 4)
    out = Sigmoid.apply(W2 @ h + b2)         # (1, 4)

    # BCE loss:  -mean( Y*log(out) + (1-Y)*log(1-out) )
    from nn.autograd.ops.log import Log
    eps_t = Tensor(np.full(out.Shape, 1e-7, dtype=np.float32))
    safe_out = out + eps_t
    safe_inv = (Tensor(np.ones(out.Shape, dtype=np.float32)) - out) + eps_t

    loss = -(Y_xor * Log.apply(safe_out) +
             (Tensor(np.ones(Y_xor.Shape, dtype=np.float32)) - Y_xor) *
             Log.apply(safe_inv)).mean()

    loss.backward()

    # SGD update
    for p in params:
        p.Data -= lr * p.Grad

    if step % 1000 == 0:
        preds = (sigmoid_np(
            sigmoid_np(W1.Data @ X_xor.Data + b1.Data)
        ) if False else
            1 / (1 + np.exp(-(W2.Data @ (1 / (1 + np.exp(-(W1.Data @ X_xor.Data + b1.Data)))) + b2.Data)))
        )
        print(f"step {step:4d}  loss={loss.Data:.4f}  "
              f"preds={np.round(preds.flatten(), 2)}")

print("Done. XOR predictions should be close to [0, 1, 1, 0]")
