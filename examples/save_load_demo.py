"""Save/Load demo — train a model, persist it, reload it, verify it still works.

Demonstrates three usage patterns:
  1. Full Save / Load           — structure + weights in one file
  2. SaveStateDict / LoadStateDict — weights only (model defined in code)
  3. Inspecting a saved file    — the format is just an npz + a JSON blob
"""

import os
import json
import numpy as np

import nn
from nn import (
    Tensor, Sequential, Linear, ReLU, Sigmoid, Softmax, MSELoss,
    Save, Load, SaveStateDict, LoadStateDict,
)
from nn.optim import Adam


HERE = os.path.dirname(__file__)
FULL_PATH = os.path.join(HERE, "xor_full.npz")
WEIGHTS_PATH = os.path.join(HERE, "xor_weights.npz")


# ---------------------------------------------------------------------------
# 1. Train a tiny XOR model to convergence
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Train a model")
print("=" * 60)

X = Tensor(np.array([[0, 0, 1, 1],
                     [0, 1, 0, 1]], dtype=np.float32))
y = Tensor(np.array([[0, 1, 1, 0]], dtype=np.float32))

model = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid(),
)
criterion = MSELoss()
optimizer = Adam(model.Parameters, lr=0.05)

for step in range(3000):
    loss = criterion(model(X), y)
    optimizer.ZeroGrad()
    loss.backward()
    optimizer.Step()

original_preds = model(X).Data.copy()
print(f"Trained loss: {float(loss.Data):.6f}")
print(f"Predictions:  {original_preds.flatten().round(3)}")


# ---------------------------------------------------------------------------
# 2. Save full model (structure + weights) and reload from scratch
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("2. Save + Load full model")
print("=" * 60)

Save(model, FULL_PATH)
print(f"Saved to {os.path.basename(FULL_PATH)} ({os.path.getsize(FULL_PATH)} bytes)")

# Note: we don't reference the original `model` here — the loaded model is
# fully reconstructed from the file alone.
reloaded = Load(FULL_PATH)
reloaded_preds = reloaded(X).Data
diff = float(np.abs(original_preds - reloaded_preds).max())
print(f"Loaded predictions: {reloaded_preds.flatten().round(3)}")
print(f"Max difference from original: {diff:.2e}  {'OK' if diff < 1e-6 else 'FAIL'}")


# ---------------------------------------------------------------------------
# 3. Inspect the saved file — it's just an npz with a JSON structure blob
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("3. Inspect saved file")
print("=" * 60)

with np.load(FULL_PATH, allow_pickle=False) as data:
    print(f"Keys in archive: {data.files}")
    structure = json.loads(str(data["_structure_"]))
    print("Structure (pretty):")
    print(json.dumps(structure, indent=2))
    print(f"\nFirst layer weight shape: {data['module_0.weight'].shape}")


# ---------------------------------------------------------------------------
# 4. SaveStateDict / LoadStateDict — weights only, model defined in code
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("4. Weights-only save/load (PyTorch-style state dict)")
print("=" * 60)

SaveStateDict(model, WEIGHTS_PATH)
print(f"Saved weights only to {os.path.basename(WEIGHTS_PATH)} "
      f"({os.path.getsize(WEIGHTS_PATH)} bytes)")

# Build a fresh model with the SAME architecture in code, then load weights
fresh = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid(),
)
print(f"Fresh model preds (before load): {fresh(X).Data.flatten().round(3)}")

LoadStateDict(fresh, WEIGHTS_PATH)
print(f"Fresh model preds (after  load): {fresh(X).Data.flatten().round(3)}")


# ---------------------------------------------------------------------------
# 5. Demonstrate strict=False for partial loads (e.g. transfer learning)
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("5. Mismatched architecture — strict vs lenient")
print("=" * 60)

bigger = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid(),
    Linear(1, 1),       # extra layer that isn't in the saved weights
    Sigmoid(),
)
try:
    LoadStateDict(bigger, WEIGHTS_PATH, strict=True)
except KeyError as e:
    print(f"strict=True raised KeyError: {e}")

LoadStateDict(bigger, WEIGHTS_PATH, strict=False)
print("strict=False loaded the matching params, left the extras at init values")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

for path in (FULL_PATH, WEIGHTS_PATH):
    os.remove(path)
print(f"\nCleaned up demo files.")
