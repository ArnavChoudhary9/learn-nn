from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Neg(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor) -> Tensor:  # type: ignore[override]
        return Tensor(-a.Data)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        return (Tensor(-grad_output.Data),)
