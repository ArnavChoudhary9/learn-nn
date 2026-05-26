import numpy as np
from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Log(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor) -> Tensor:  # type: ignore[override]
        ctx.SaveForBackward(a)
        return Tensor(np.log(a.Data).astype(np.float32))

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        (a,) = ctx.SavedTensors
        return (Tensor(grad_output.Data / a.Data),)
