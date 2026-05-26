import numpy as np
from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Mul(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor, b: Tensor | float | int) -> Tensor:  # type: ignore[override]
        b_data = b.Data if isinstance(b, Tensor) else np.asarray(b, dtype=np.float32)
        ctx.SaveForBackward(a, b_data)
        return Tensor(a.Data * b_data)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        a, b_data = ctx.SavedTensors[0], ctx.SavedTensors[1]
        g = grad_output.Data
        return (Tensor(g * b_data), Tensor(g * a.Data))
