"""Tensor primitive."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..autograd.graph import Node


class Tensor:
    """Tensor primitive."""

    _Data: np.ndarray
    _Grad: np.ndarray | None
    _GradFn: Node | None
    _IsLeaf: bool

    _RequiresGrad: bool = False

    def __init__(self, data: np.ndarray | list, requiresGrad: bool = False):
        self._Data = np.asarray(data, dtype=np.float32)
        self._RequiresGrad = requiresGrad
        self._GradFn = None
        self._IsLeaf = True

        self._Grad = (
            np.zeros(self._Data.shape, dtype=np.float32)
            if requiresGrad
            else None
        )

    @property
    def Data(self) -> np.ndarray:
        """Data."""
        return self._Data
    
    @Data.setter
    def Data(self, value: np.ndarray):
        """Set data."""
        self._Data = value

    @property
    def Grad(self) -> np.ndarray | None:
        """Gradient."""
        return self._Grad

    @Grad.setter
    def Grad(self, value: np.ndarray | None):
        """Set gradient."""
        self._Grad = value

    @property
    def RequiresGrad(self) -> bool:
        """Whether gradient is required."""
        return self._RequiresGrad
    
    @property
    def Shape(self) -> tuple[int, ...]:
        """Shape."""
        return self._Data.shape

    def backward(self, grad=None):
        from ..autograd.engine import backward
        backward(self, grad)

    # --- operator overloads ---

    def __add__(self, other):
        from ..autograd.ops.add import Add
        return Add.apply(self, other)

    def __radd__(self, other):
        from ..autograd.ops.add import Add
        return Add.apply(self, other)

    def __sub__(self, other):
        from ..autograd.ops.sub import Sub
        return Sub.apply(self, other)

    def __rsub__(self, other):
        from ..autograd.ops.sub import Sub
        return Sub.apply(other, self)

    def __neg__(self):
        from ..autograd.ops.neg import Neg
        return Neg.apply(self)

    def __mul__(self, other):
        from ..autograd.ops.mul import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other):
        from ..autograd.ops.mul import Mul
        return Mul.apply(self, other)

    def __truediv__(self, other):
        from ..autograd.ops.div import Div
        return Div.apply(self, other)

    def __matmul__(self, other):
        from ..autograd.ops.matmul import MatMul
        return MatMul.apply(self, other)

    def __pow__(self, exponent):
        from ..autograd.ops.pow import Pow
        return Pow.apply(self, exponent)

    def sum(self):
        from ..autograd.ops.sum import Sum
        return Sum.apply(self)

    def mean(self):
        from ..autograd.ops.mean import Mean
        return Mean.apply(self)
