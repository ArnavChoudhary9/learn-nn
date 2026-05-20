"""XOR training example."""

from nn import *

import numpy as np

# Define the XOR dataset
X = Tensor(
    np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ])
)
y = Tensor(
    np.array([
        [0, 1, 1, 0],
    ])
)

# Define a simple feedforward neural network
model = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid()
)

# Define the loss function and optimizer
criterion = MSELoss()
optimizer = SGD(model.Parameters, lr=0.1)

# Training loop
for epoch in range(100_000):
    # Forward pass
    outputs = model.Forward(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.ZeroGrad()
    dY = criterion.Backward()
    model.Backward(dY)
    optimizer.Step()

    if (epoch + 1) % 5000 == 0:
        print(f'Epoch [{epoch + 1}/100000], Loss: {loss.Data:.4f}')

# Test the trained model
with np.printoptions(precision=4, suppress=True):
    print("Predicted outputs:")
    print(model.Forward(X).Data)
