"""Flatten op — reshapes (N, C, H, W) into (C*H*W, N) so it feeds Linear's layout."""

from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class Flatten(Function):
    @classmethod
    def Forward(cls, ctx: Context, x: Tensor) -> Tensor:  # type: ignore[override]
        x_data = x.Data
        N = x_data.shape[0]
        ctx.SaveForBackward(x_data.shape)
        return Tensor(x_data.reshape(N, -1).T)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        (orig_shape,) = ctx.SavedTensors
        # grad_output: (features, N) -> (N, features) -> (N, C, H, W)
        grad = grad_output.Data.T.reshape(orig_shape)
        return (Tensor(grad),)
