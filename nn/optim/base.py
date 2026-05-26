"""Optimizer base class."""

import numpy as np

from ..core.parameter import Parameter


class Optimizer:
    """Common interface: holds parameters, exposes Step/ZeroGrad."""

    _Parameters: list[Parameter]
    _LearningRate: float

    def __init__(self, parameters: list[Parameter], lr: float):
        self._Parameters = parameters
        self._LearningRate = lr

    def ZeroGrad(self) -> None:
        for p in self._Parameters:
            if p.Grad is not None:
                p.Grad.fill(0.0)
            else:
                p.Grad = np.zeros(p.Data.shape, dtype=np.float32)

    def Step(self) -> None:
        raise NotImplementedError
