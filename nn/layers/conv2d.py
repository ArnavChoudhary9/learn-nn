"""2D convolutional layer."""

from typing import Callable
import numpy as np

from ..core.tensor import Tensor
from ..core.parameter import Parameter
from ..core.module import Module
from ..autograd.ops.conv2d import Conv2D as Conv2DOp


def _DefaultConvInit(shape: tuple[int, ...]) -> np.ndarray:
    """He-style init using the true conv fan_in = C_in * K * K."""
    _, C_in, K, _ = shape
    fan_in = C_in * K * K
    limit = float(np.sqrt(2.0 / fan_in))
    return np.random.uniform(-limit, limit, size=shape)


class Conv2D(Module):
    """2D convolutional layer (NCHW). No padding."""

    _InputChannels: int
    _OutputChannels: int
    _KernelSize: int
    _Stride: int

    def __init__(
        self,
        inputChannels: int,
        outputChannels: int,
        kernelSize: int,
        stride: int = 1,
        initMethod: Callable[..., np.ndarray] = _DefaultConvInit,
        *initArgs,
        **initKwargs,
    ) -> None:
        super().__init__()
        self._InputChannels = inputChannels
        self._OutputChannels = outputChannels
        self._KernelSize = kernelSize
        self._Stride = stride

        weightShape = (outputChannels, inputChannels, kernelSize, kernelSize)
        weightData = initMethod(weightShape, *initArgs, **initKwargs)
        biasData = np.zeros((outputChannels,), dtype=np.float32)

        self.AddParameter("weight", Parameter(weightData))
        self.AddParameter("bias", Parameter(biasData))

    def Forward(self, x: Tensor) -> Tensor:
        weight = self._Parameters["weight"]
        bias = self._Parameters["bias"]
        return Conv2DOp.apply(x, weight, bias, self._Stride)

    def Config(self) -> dict:
        return {
            "type": "Conv2D",
            "args": {
                "inputChannels": self._InputChannels,
                "outputChannels": self._OutputChannels,
                "kernelSize": self._KernelSize,
                "stride": self._Stride,
            },
        }
