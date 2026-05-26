"""Softmax activation."""

from ..core.tensor import Tensor
from ..core.module import Module
from ..autograd.ops.softmax import Softmax as SoftmaxOp


class Softmax(Module):
    """Softmax along the given axis (default 0 — class axis for (C, B) tensors)."""

    _Axis: int

    def __init__(self, axis: int = 0):
        super().__init__()
        self._Axis = axis

    def Forward(self, x: Tensor) -> Tensor:
        return SoftmaxOp.apply(x, self._Axis)

    def Config(self) -> dict:
        return {"type": "Softmax", "args": {"axis": self._Axis}}
