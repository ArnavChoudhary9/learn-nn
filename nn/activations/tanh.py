"""Tanh activation."""

from ..core.tensor import Tensor
from ..core.module import Module
from ..autograd.ops.tanh import Tanh as TanhOp


class Tanh(Module):
    def Forward(self, x: Tensor) -> Tensor:
        return TanhOp.apply(x)
