"""Base Module class."""

from abc import ABC, abstractmethod

from .tensor import Tensor
from .parameter import Parameter

class Module(ABC):
    """Base Module class."""

    _Parameters: dict[str, Parameter]
    _Modules: dict[str, 'Module']

    def __init__(self):
        self._Parameters = {}
        self._Modules = {}

    @property
    def Parameters(self) -> list[Parameter]:
        """Return a list of all parameters in the module."""
        params = []
        for _, param in self._Parameters.items():
            params.append(param)
        for _, module in self._Modules.items():
            params.extend(module.Parameters)
        return params

    def AddParameter(self, name: str, param: Parameter):
        """Add a parameter to the module."""
        self._Parameters[name] = param

    def AddModule(self, name: str, module: 'Module'):
        """Add a submodule to the module."""
        self._Modules[name] = module

    @abstractmethod
    def Forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        pass

    @abstractmethod
    def Backward(self, dZ: Tensor) -> Tensor:
        """Backward pass."""
        pass
