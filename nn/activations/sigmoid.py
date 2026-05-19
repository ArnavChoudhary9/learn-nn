"""Sigmoid activation."""

from ..core.tensor import Tensor
from ..core.module import Module

import numpy as np

class Sigmoid(Module):
    """Sigmoid activation."""

    _TemporaryOutput: Tensor | None

    def __init__(self):
        super().__init__()
        self._TemporaryOutput = None

    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        sigmoidOutput = 1 / (1 + np.exp(-x.Data))
        self._TemporaryOutput = Tensor(sigmoidOutput)
        return self._TemporaryOutput
    
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        if self._TemporaryOutput is None:
            raise ValueError("No output to backward pass.")
        
        sigmoidOutput = self._TemporaryOutput.Data
        dX = dZ.Data * sigmoidOutput * (1 - sigmoidOutput)
        return Tensor(dX)
