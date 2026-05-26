"""Base Module class."""

from abc import ABC, abstractmethod

import numpy as np

from .tensor import Tensor
from .parameter import Parameter


class Module(ABC):
    """Base Module class.

    With autograd, modules only need to implement `Forward`. Gradients flow
    automatically through the Tensor operations performed inside `Forward`,
    so there is no manual `Backward` method on subclasses anymore.
    """

    _Parameters: dict[str, Parameter]
    _Modules: dict[str, 'Module']

    def __init__(self):
        self._Parameters = {}
        self._Modules = {}

    @property
    def Parameters(self) -> list[Parameter]:
        """All parameters in this module and its submodules (flat list)."""
        params: list[Parameter] = []
        for param in self._Parameters.values():
            params.append(param)
        for module in self._Modules.values():
            params.extend(module.Parameters)
        return params

    def AddParameter(self, name: str, param: Parameter) -> None:
        self._Parameters[name] = param

    def AddModule(self, name: str, module: 'Module') -> None:
        self._Modules[name] = module

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.Forward(*args, **kwargs)

    @abstractmethod
    def Forward(self, x: Tensor) -> Tensor:
        ...

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def StateDict(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Collect parameters into a flat dict keyed by hierarchical name."""
        sd: dict[str, np.ndarray] = {}
        for name, p in self._Parameters.items():
            sd[f"{prefix}{name}"] = p.Data
        for name, m in self._Modules.items():
            sd.update(m.StateDict(prefix=f"{prefix}{name}."))
        return sd

    def LoadStateDict(self, sd: dict[str, np.ndarray], prefix: str = "", strict: bool = True) -> None:
        """Load parameters from a flat dict."""
        for name, p in self._Parameters.items():
            key = f"{prefix}{name}"
            if key not in sd:
                if strict:
                    raise KeyError(f"Missing parameter in state dict: {key}")
                continue
            arr = np.asarray(sd[key], dtype=np.float32)
            if arr.shape != p.Data.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: expected {p.Data.shape}, got {arr.shape}"
                )
            p.Data = arr
        for name, m in self._Modules.items():
            m.LoadStateDict(sd, prefix=f"{prefix}{name}.", strict=strict)

    def Config(self) -> dict:
        """Return a dict describing how to reconstruct this module.

        Default: just the class name (works for modules with no init args
        like ReLU, Sigmoid, Tanh). Subclasses with constructor args must
        override and include them under `"args"`.
        """
        return {"type": type(self).__name__}
