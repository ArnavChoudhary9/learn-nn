import numpy as np
from ..function import Function
from ..context import Context
from ...core.tensor import Tensor
from .utils import UnbroadcastGrad


class Add(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor, b: Tensor | float | int) -> Tensor:  # type: ignore[override]
        a_data = a.Data if isinstance(a, Tensor) else np.asarray(a, dtype=np.float32)
        b_data = b.Data if isinstance(b, Tensor) else np.asarray(b, dtype=np.float32)
        ctx.SaveForBackward(a_data.shape, b_data.shape)
        return Tensor(a_data + b_data)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        a_shape, b_shape = ctx.SavedTensors[0], ctx.SavedTensors[1]
        g = grad_output.Data
        return (
            Tensor(UnbroadcastGrad(g, a_shape)),
            Tensor(UnbroadcastGrad(g, b_shape)),
        )
