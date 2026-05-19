"""Tanh activation."""

from ..core.tensor import Tensor
from ..core.module import Module

import numpy as np

class Tanh(Module):
    """Tanh activation."""

    _TemporaryInput: Tensor | None

    def __init__(self):
        super().__init__()
        self._TemporaryInput = None

    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        tanhOutput = np.tanh(x.Data)
        self._TemporaryInput = Tensor(tanhOutput)
        return Tensor(tanhOutput)
    
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        if self._TemporaryInput is None:
            raise ValueError("No input to backward pass.")
        
        tanhOutput = self._TemporaryInput.Data
        dX = dZ.Data * (1 - tanhOutput ** 2)
        return Tensor(dX)
