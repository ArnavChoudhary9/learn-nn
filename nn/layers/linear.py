"""Linear (fully-connected) layer."""

from ..core.tensor import Tensor
from ..core.parameter import Parameter
from ..core.module import Module

import numpy as np

class Linear(Module):
    """Linear (fully-connected) layer."""

    _InputDim: int
    _OutputDim: int

    _TemporaryInput: Tensor | None

    def __init__(self, inputDim: int, outputDim: int):
        super().__init__()
        self._InputDim = inputDim
        self._OutputDim = outputDim
        self._TemporaryInput = None

        # Initialize weights and biases
        weightData = np.random.randn(outputDim, inputDim) * np.sqrt(2. / inputDim)
        biasData = np.zeros((outputDim, 1))

        self.AddParameter('weight', Parameter(weightData))
        self.AddParameter('bias', Parameter(biasData))

    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        weight = self._Parameters['weight'].Data
        bias = self._Parameters['bias'].Data

        assert x.Data.shape[0] == weight.shape[1], (
            f"Expected input features "
            f"{weight.shape[1]}, "
            f"got {x.Data.shape[0]}"
        )
        
        self._TemporaryInput = x
        return Tensor(weight @ x.Data + bias)
    
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        if self._TemporaryInput is None:
            raise ValueError("No input to backward pass.")
        
        _dZ = dZ.Data
        W = self._Parameters['weight'].Data
        # B = self._Parameters['bias'].Data
        X = self._TemporaryInput.Data

        # Compute gradients
        dW = _dZ @ X.T
        dB = np.sum(_dZ, axis=1, keepdims=True)
        dX = W.T @ _dZ

        # Grad accumulation (if needed)
        # dW += _dZ @ X.T
        # dB += np.sum(_dZ, axis=1, keepdims=True)
        # dX += W.T @ _dZ

        # Clear temporary input
        # self._TemporaryInput = None   # Uncomment if one backward per forward guaranteed

        # Update parameters
        self._Parameters['weight'].Grad = dW
        self._Parameters['bias'].Grad = dB

        return Tensor(dX)
