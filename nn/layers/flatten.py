"""Flatten layer — (N, C, H, W) -> (C*H*W, N) to feed Linear's column-major layout."""

from ..core.tensor import Tensor
from ..core.module import Module
from ..autograd.ops.flatten import Flatten as FlattenOp


class Flatten(Module):
    """Reshape an NCHW input into the (features, batch) layout Linear expects."""

    def Forward(self, x: Tensor) -> Tensor:
        return FlattenOp.apply(x)
