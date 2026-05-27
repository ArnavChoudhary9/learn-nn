"""2D max-pooling layer."""

from ..core.tensor import Tensor
from ..core.module import Module
from ..autograd.ops.maxpool2d import MaxPool2D as MaxPool2DOp


class MaxPool2D(Module):
    """Max-pool over K×K windows. No padding."""

    _KernelSize: int
    _Stride: int

    def __init__(self, kernelSize: int, stride: int | None = None) -> None:
        super().__init__()
        self._KernelSize = kernelSize
        self._Stride = stride if stride is not None else kernelSize

    def Forward(self, x: Tensor) -> Tensor:
        return MaxPool2DOp.apply(x, self._KernelSize, self._Stride)

    def Config(self) -> dict:
        return {
            "type": "MaxPool2D",
            "args": {"kernelSize": self._KernelSize, "stride": self._Stride},
        }
