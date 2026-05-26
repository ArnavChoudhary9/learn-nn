import numpy as np
from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Tanh(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor) -> Tensor:  # type: ignore[override]
        out = np.tanh(a.Data).astype(np.float32)
        ctx.SaveForBackward(out)
        return Tensor(out)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        (out,) = ctx.SavedTensors
        return (Tensor(grad_output.Data * (1.0 - out * out)),)
