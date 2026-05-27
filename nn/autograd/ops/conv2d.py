"""2D convolution op.

Input convention: NCHW.
  x:      (N, C_in, H, W)
  weight: (C_out, C_in, K, K)
  bias:   (C_out,)

Forward uses an im2col-style rearrangement so the conv reduces to one matmul.
"""

import numpy as np

from ..function import Function
from ..context import Context
from ...core.tensor import Tensor


def _Im2Col(x: np.ndarray, k: int, s: int) -> tuple[np.ndarray, int, int]:
    """Pull every K×K patch into an explicit (N, C, K, K, H_out, W_out) array."""
    N, C, H, W = x.shape
    H_out = (H - k) // s + 1
    W_out = (W - k) // s + 1
    cols = np.zeros((N, C, k, k, H_out, W_out), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j, :, :] = x[:, :, i:i + s * H_out:s, j:j + s * W_out:s]
    return cols, H_out, W_out


def _Col2Im(cols: np.ndarray, x_shape: tuple[int, ...], k: int, s: int) -> np.ndarray:
    """Inverse of _Im2Col — accumulates overlapping patch gradients back into x."""
    grad_x = np.zeros(x_shape, dtype=cols.dtype)
    H_out = cols.shape[4]
    W_out = cols.shape[5]
    for i in range(k):
        for j in range(k):
            grad_x[:, :, i:i + s * H_out:s, j:j + s * W_out:s] += cols[:, :, i, j, :, :]
    return grad_x


class Conv2D(Function):
    @classmethod
    def Forward(  # type: ignore[override]
        cls,
        ctx: Context,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: int = 1,
    ) -> Tensor:
        x_data = x.Data
        w_data = weight.Data
        b_data = bias.Data

        N, C_in, H, W = x_data.shape
        C_out, _, K, _ = w_data.shape

        cols, H_out, W_out = _Im2Col(x_data, K, stride)
        # (C_in*K*K, N*H_out*W_out)
        cols_2d = cols.transpose(1, 2, 3, 0, 4, 5).reshape(C_in * K * K, N * H_out * W_out)
        w_2d = w_data.reshape(C_out, C_in * K * K)

        out_2d = w_2d @ cols_2d  # (C_out, N*H_out*W_out)
        out = out_2d.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)
        out = out + b_data.reshape(1, C_out, 1, 1)

        ctx.SaveForBackward(cols_2d, w_2d, x_data.shape, K, stride, H_out, W_out)
        return Tensor(out)

    @classmethod
    def Backward(cls, ctx: Context, grad_output: Tensor) -> tuple:  # type: ignore[override]
        cols_2d, w_2d, x_shape, K, stride, H_out, W_out = ctx.SavedTensors
        N, C_in, _, _ = x_shape
        C_out = w_2d.shape[0]

        g = grad_output.Data  # (N, C_out, H_out, W_out)
        g_2d = g.transpose(1, 0, 2, 3).reshape(C_out, N * H_out * W_out)

        grad_w_2d = g_2d @ cols_2d.T  # (C_out, C_in*K*K)
        grad_w = grad_w_2d.reshape(C_out, C_in, K, K)

        grad_b = g.sum(axis=(0, 2, 3))  # (C_out,)

        grad_cols_2d = w_2d.T @ g_2d  # (C_in*K*K, N*H_out*W_out)
        grad_cols = grad_cols_2d.reshape(C_in, K, K, N, H_out, W_out).transpose(3, 0, 1, 2, 4, 5)
        grad_x = _Col2Im(grad_cols, x_shape, K, stride)

        return (Tensor(grad_x), Tensor(grad_w), Tensor(grad_b), None)
