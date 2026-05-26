import numpy as np
from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Softmax(Function):
    @classmethod
    def Forward(cls, ctx: Context, a: Tensor, axis: int = 0) -> Tensor:  # type: ignore[override]
        shifted = a.Data - np.max(a.Data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out = (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32)
        ctx.SaveForBackward(out, axis)
        return Tensor(out)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        out, axis = ctx.SavedTensors[0], ctx.SavedTensors[1]
        g = grad_output.Data
        # dx_i = S_i * (dZ_i - sum_axis(dZ * S))
        dx = out * (g - np.sum(g * out, axis=axis, keepdims=True))
        return (Tensor(dx), None)
