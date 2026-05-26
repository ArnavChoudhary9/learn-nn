"""Base Module class."""

from abc import ABC, abstractmethod

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
        """Return a list of all parameters in this module and its submodules."""
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
        """Forward pass — must use Tensor ops so autograd can track gradients."""
        ...
