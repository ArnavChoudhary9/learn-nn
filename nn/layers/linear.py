"""Linear (fully-connected) layer."""

from typing import Callable
import numpy as np

from ..core.tensor import Tensor
from ..core.parameter import Parameter
from ..core.module import Module
from ..init.xavier import XavierInitialization
from ..autograd.ops.matmul import MatMul
from ..autograd.ops.add import Add


class Linear(Module):
    """Linear (fully-connected) layer: y = W @ x + b."""

    _InputDim: int
    _OutputDim: int

    def __init__(
        self,
        inputDim: int,
        outputDim: int,
        initMethod: Callable[..., np.ndarray] = XavierInitialization,
        *initArgs,
        **initKwargs,
    ) -> None:
        super().__init__()
        self._InputDim = inputDim
        self._OutputDim = outputDim

        weightData = initMethod((outputDim, inputDim), *initArgs, **initKwargs)
        biasData = np.zeros((outputDim, 1), dtype=np.float32)

        self.AddParameter("weight", Parameter(weightData))
        self.AddParameter("bias", Parameter(biasData))

    def Forward(self, x: Tensor) -> Tensor:
        weight = self._Parameters["weight"]
        bias = self._Parameters["bias"]
        return Add.apply(MatMul.apply(weight, x), bias)
