from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Pow(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor, n: float | int) -> Tensor:  # type: ignore[override]
        ctx.SaveForBackward(a, n)
        return Tensor(a.Data ** n)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        a, n = ctx.SavedTensors[0], ctx.SavedTensors[1]
        return (Tensor(grad_output.Data * n * a.Data ** (n - 1)), None)
