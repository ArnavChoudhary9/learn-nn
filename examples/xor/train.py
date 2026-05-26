"""XOR training — fully autograd-driven."""

import numpy as np

from nn import Tensor, Sequential, Linear, Sigmoid, MSELoss
from nn.optim import Adam


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

EPOCHS = 5_000
for epoch in range(1, EPOCHS + 1):
    out = model(X)
    loss = criterion(out, y)

    optimizer.ZeroGrad()
    loss.backward()
    optimizer.Step()

    if epoch % 500 == 0:
        print(f"epoch [{epoch:5d}/{EPOCHS}]  loss: {float(loss.Data):.6f}")

with np.printoptions(precision=4, suppress=True):
    print("\nPredictions:")
    print(model(X).Data)
    print("Expected:")
    print(y.Data)
