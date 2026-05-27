"""Sequential container."""

from ..core.tensor import Tensor
from ..core.module import Module


class Sequential(Module):
    """Chain modules left-to-right. Autograd handles backprop automatically.

    `inputShape` optionally pins the model's external contract — the full
    expected input shape with `None` marking the batch axis. When set, it
    survives Save/Load so callers can feed the model without knowing whether
    it's an MLP `(features, None)` or a CNN `(None, C, H, W)`.
    """

    _InputShape: tuple | None

    def __init__(self, *modules: Module, inputShape: tuple | None = None):
        super().__init__()
        self._InputShape = tuple(inputShape) if inputShape is not None else None
        for i, module in enumerate(modules):
            self.AddModule(f"module_{i}", module)

    def Forward(self, x: Tensor) -> Tensor:
        for module in self._Modules.values():
            x = module(x)
        return x

    @property
    def InputShape(self) -> tuple | None:
        return self._InputShape

    def Config(self) -> dict:
        cfg: dict = {
            "type": "Sequential",
            "modules": [m.Config() for m in self._Modules.values()],
        }
        if self._InputShape is not None:
            # JSON serializes None as null, round-trips back to None.
            cfg["inputShape"] = list(self._InputShape)
        return cfg
