"""ReLU activation."""

from ..core.tensor import Tensor
from ..core.module import Module
from ..autograd.ops.relu import ReLU as ReLUOp


class ReLU(Module):
    def Forward(self, x: Tensor) -> Tensor:
        return ReLUOp.apply(x)
