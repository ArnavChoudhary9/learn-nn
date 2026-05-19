"""Softmax activation."""

from ..core.tensor import Tensor
from ..core.module import Module

import numpy as np

class Softmax(Module):
    """Softmax activation."""

    _TemporaryInput: Tensor | None

    def __init__(self):
        super().__init__()
        self._TemporaryInput = None

    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Shift input for numerical stability
        shiftedX = x.Data - np.max(x.Data, axis=0, keepdims=True)
        expX = np.exp(shiftedX)
        softmaxOutput = expX / np.sum(expX, axis=0, keepdims=True)
        self._TemporaryInput = Tensor(softmaxOutput)
        return Tensor(softmaxOutput)
    
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        if self._TemporaryInput is None:
            raise ValueError("No input to backward pass.")
        
        softmaxOutput = self._TemporaryInput.Data
        dX = softmaxOutput * (dZ.Data - np.sum(dZ.Data * softmaxOutput, axis=0, keepdims=True))
        return Tensor(dX)
