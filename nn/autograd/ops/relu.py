from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class ReLU(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor) -> Tensor:  # type: ignore[override]
        mask = a.Data > 0
        ctx.SaveForBackward(mask)
        return Tensor(a.Data * mask)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        (mask,) = ctx.SavedTensors
        return (Tensor(grad_output.Data * mask),)
