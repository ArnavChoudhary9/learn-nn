"""ReLU activation."""

from ..core.tensor import Tensor
from ..core.module import Module

import numpy as np

class ReLU(Module):
    """ReLU activation."""

    _TemporaryInput: Tensor | None

    def __init__(self):
        super().__init__()
        self._TemporaryInput = None

    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        reluOutput = np.maximum(0, x.Data)
        self._TemporaryInput = x
        return Tensor(reluOutput)
    
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        if self._TemporaryInput is None:
            raise ValueError("No input to backward pass.")
        
        reluGrad = (self._TemporaryInput.Data > 0).astype(float)
        dX = dZ.Data * reluGrad
        return Tensor(dX)
