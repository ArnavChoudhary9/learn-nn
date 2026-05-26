from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class MatMul(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor, b: Tensor) -> Tensor:  # type: ignore[override]
        ctx.SaveForBackward(a, b)
        return Tensor(a.Data @ b.Data)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        a, b = ctx.SavedTensors[0], ctx.SavedTensors[1]
        g = grad_output.Data
        return (Tensor(g @ b.Data.T), Tensor(a.Data.T @ g))
