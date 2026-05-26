"""Base class for differentiable operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from .context import Context
from .graph import Node

if TYPE_CHECKING:
    from ..core.tensor import Tensor


class Function:

    @classmethod
    def apply(cls, *inputs: Tensor | float | int | np.ndarray) -> Tensor:
        """Run forward pass, build graph node, return output Tensor."""
        from ..core.tensor import Tensor

        ctx = Context()
        out_tensor = cls.Forward(ctx, *inputs)  # type: ignore[call-arg]

        requires_grad = any(
            isinstance(inp, Tensor) and inp.RequiresGrad for inp in inputs
        )

        # Forward returns a plain Tensor (no grad tracking); re-wrap with tracking.
        result = Tensor(out_tensor.Data, requiresGrad=requires_grad)
        if requires_grad:
            result._GradFn = Node(cls, ctx, list(inputs))
            result._IsLeaf = False

        return result

    @classmethod
    def Forward(cls, ctx: Context, *args: object) -> Tensor:
        raise NotImplementedError

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple:
        raise NotImplementedError
