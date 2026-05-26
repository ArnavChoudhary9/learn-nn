"""Sigmoid activation."""

from ..core.tensor import Tensor
from ..core.module import Module
from ..autograd.ops.sigmoid import Sigmoid as SigmoidOp


class Sigmoid(Module):
    def Forward(self, x: Tensor) -> Tensor:
        return SigmoidOp.apply(x)
