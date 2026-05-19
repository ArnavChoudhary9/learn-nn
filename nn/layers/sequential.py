"""Sequential container."""

from ..core.tensor import Tensor
from ..core.parameter import Parameter
from ..core.module import Module

import numpy as np

class Sequential(Module):
    """Sequential container."""

    _Modules: dict[str, Module]

    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.AddModule(f'module_{i}', module)

    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for _, module in self._Modules.items():
            x = module.Forward(x)
        return x
    
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        for _, module in reversed(self._Modules.items()):
            dZ = module.Backward(dZ)
        return dZ
