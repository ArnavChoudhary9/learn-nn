"""2D max-pooling op (NCHW, no padding)."""

import numpy as np

from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


class MaxPool2D(Function):
    @classmethod
    def Forward(  # type: ignore[override]
        cls,
        ctx: Context,
        x: Tensor,
        kernelSize: int,
        stride: int,
    ) -> Tensor:
        x_data = x.Data
        N, C, H, W = x_data.shape
        k, s = kernelSize, stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        patches = np.empty((N, C, k * k, H_out, W_out), dtype=x_data.dtype)
        for i in range(k):
            for j in range(k):
                patches[:, :, i * k + j, :, :] = x_data[
                    :, :, i:i + s * H_out:s, j:j + s * W_out:s
                ]

        argmax = patches.argmax(axis=2)  # (N, C, H_out, W_out)
        out = np.take_along_axis(patches, argmax[:, :, None, :, :], axis=2).squeeze(2)

        ctx.SaveForBackward(argmax, x_data.shape, k, s)
        return Tensor(out)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple:  # type: ignore[override]
        argmax, x_shape, k, s = ctx.SavedTensors
        H_out, W_out = grad_output.Data.shape[-2:]
        grad_x = np.zeros(x_shape, dtype=grad_output.Data.dtype)

        g = grad_output.Data
        for i in range(k):
            for j in range(k):
                mask = (argmax == i * k + j).astype(g.dtype)
                grad_x[:, :, i:i + s * H_out:s, j:j + s * W_out:s] += g * mask

        return (Tensor(grad_x), None, None)
