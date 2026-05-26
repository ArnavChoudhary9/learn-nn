"""Speed comparison: autograd-based training vs hand-coded numpy backprop.

Both implementations train an identical 3-layer MLP (input -> 128 -> 64 -> 10)
on synthetic data for N steps with the same initial weights and SGD.
We compare wall-clock time and confirm both reach the same final loss.
"""

import time
import numpy as np

from nn import Tensor, Sequential, Linear, ReLU, Softmax, CELoss
from nn.optim import SGD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IN_DIM   = 200
H1, H2   = 128, 64
OUT_DIM  = 10
BATCH    = 128
STEPS    = 200
LR       = 0.05
SEED     = 0


# ---------------------------------------------------------------------------
# Shared data + shared initial weights so both runs start from the same point
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
X_np = rng.standard_normal((IN_DIM, BATCH)).astype(np.float32)
labels = rng.integers(0, OUT_DIM, BATCH)
Y_np = np.zeros((OUT_DIM, BATCH), dtype=np.float32)
Y_np[labels, np.arange(BATCH)] = 1.0


def he(shape):
    fan_in = shape[1]
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


W1_init = he((H1, IN_DIM));  b1_init = np.zeros((H1, 1), dtype=np.float32)
W2_init = he((H2, H1));      b2_init = np.zeros((H2, 1), dtype=np.float32)
W3_init = he((OUT_DIM, H2)); b3_init = np.zeros((OUT_DIM, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# 1) Autograd version — uses our Tensor + autograd ops
# ---------------------------------------------------------------------------

def run_autograd():
    model = Sequential(
        Linear(IN_DIM, H1), ReLU(),
        Linear(H1, H2),     ReLU(),
        Linear(H2, OUT_DIM), Softmax(),
    )
    # Inject identical initial weights
    params = model.Parameters
    params[0].Data = W1_init.copy(); params[1].Data = b1_init.copy()
    params[2].Data = W2_init.copy(); params[3].Data = b2_init.copy()
    params[4].Data = W3_init.copy(); params[5].Data = b3_init.copy()

    loss_fn = CELoss()
    opt = SGD(model.Parameters, lr=LR)

    X = Tensor(X_np); Y = Tensor(Y_np)

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(STEPS):
        out = model(X)
        loss = loss_fn(out, Y)
        opt.ZeroGrad()
        loss.backward()
        opt.Step()
        last_loss = float(loss.Data)
    elapsed = time.perf_counter() - t0
    return elapsed, last_loss


# ---------------------------------------------------------------------------
# 2) Manual numpy backprop — same arithmetic, no autograd overhead
# ---------------------------------------------------------------------------

def run_manual():
    W1 = W1_init.copy(); b1 = b1_init.copy()
    W2 = W2_init.copy(); b2 = b2_init.copy()
    W3 = W3_init.copy(); b3 = b3_init.copy()

    X, Y = X_np, Y_np
    B = X.shape[1]

    t0 = time.perf_counter()
    last_loss = 0.0
    for _ in range(STEPS):
        # Forward
        z1 = W1 @ X + b1
        a1 = np.maximum(z1, 0)
        z2 = W2 @ a1 + b2
        a2 = np.maximum(z2, 0)
        z3 = W3 @ a2 + b3
        # softmax
        z3 -= z3.max(axis=0, keepdims=True)
        e = np.exp(z3)
        p = e / e.sum(axis=0, keepdims=True)
        # CE loss (same definition as nn.CELoss with +eps)
        eps = 1e-7
        last_loss = float(-(Y * np.log(p + eps)).sum() / B)

        # Backward (fused softmax+CE gradient: dZ3 = (p - Y) / B)
        dZ3 = (p - Y) / B
        dW3 = dZ3 @ a2.T
        db3 = dZ3.sum(axis=1, keepdims=True)
        dA2 = W3.T @ dZ3
        dZ2 = dA2 * (z2 > 0)
        dW2 = dZ2 @ a1.T
        db2 = dZ2.sum(axis=1, keepdims=True)
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (z1 > 0)
        dW1 = dZ1 @ X.T
        db1 = dZ1.sum(axis=1, keepdims=True)

        # SGD update
        W1 -= LR * dW1; b1 -= LR * db1
        W2 -= LR * dW2; b2 -= LR * db2
        W3 -= LR * dW3; b3 -= LR * db3

    elapsed = time.perf_counter() - t0
    return elapsed, last_loss


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Network: {IN_DIM} -> {H1} -> {H2} -> {OUT_DIM}")
    print(f"Batch={BATCH}  Steps={STEPS}  LR={LR}\n")

    # Warmup (JIT-y numpy paths / first-call import costs)
    run_autograd(); run_manual()

    ag_time, ag_loss = run_autograd()
    mn_time, mn_loss = run_manual()

    print(f"{'mode':<12} {'wall (s)':>10} {'final loss':>14} {'ms/step':>10}")
    print("-" * 50)
    print(f"{'autograd':<12} {ag_time:>10.3f} {ag_loss:>14.6f} {1000*ag_time/STEPS:>10.2f}")
    print(f"{'manual':<12} {mn_time:>10.3f} {mn_loss:>14.6f} {1000*mn_time/STEPS:>10.2f}")
    print()
    overhead_pct = 100.0 * (ag_time - mn_time) / mn_time
    print(f"Autograd overhead vs manual: {overhead_pct:+.1f}%  "
          f"({ag_time/mn_time:.2f}x slower)")
    print(f"Final loss match: {abs(ag_loss - mn_loss):.2e}")
