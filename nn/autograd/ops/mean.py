import numpy as np
from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Mean(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor) -> Tensor:  # type: ignore[override]
        ctx.SaveForBackward(a.Shape, a.Data.size)
        return Tensor(np.asarray(np.mean(a.Data), dtype=np.float32))

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        shape, n = ctx.SavedTensors[0], ctx.SavedTensors[1]
        return (Tensor(np.broadcast_to(grad_output.Data / n, shape).copy()),)
