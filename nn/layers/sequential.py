"""Sequential container."""

from ..core.tensor import Tensor
from ..core.module import Module


class Sequential(Module):
    """Chain modules left-to-right. Autograd handles backprop automatically."""

    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.AddModule(f"module_{i}", module)

    def Forward(self, x: Tensor) -> Tensor:
        for module in self._Modules.values():
            x = module(x)
        return x
