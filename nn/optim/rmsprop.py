"""RMSprop optimizer."""

import numpy as np

from ..core.parameter import Parameter
from .base import Optimizer


class RMSprop(Optimizer):
    """v <- rho*v + (1-rho)*g^2 ; theta <- theta - lr * g / (sqrt(v) + eps)"""

    _Rho: float
    _Eps: float
    _SquareAvg: list[np.ndarray]

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 0.001,
        rho: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__(parameters, lr)
        self._Rho = rho
        self._Eps = eps
        self._SquareAvg = [np.zeros_like(p.Data, dtype=np.float32) for p in parameters]

    def Step(self) -> None:
        for i, p in enumerate(self._Parameters):
            if p.Grad is None:
                continue
            g = p.Grad
            self._SquareAvg[i] = self._Rho * self._SquareAvg[i] + (1.0 - self._Rho) * g * g
            p.Data = p.Data - self._LearningRate * g / (np.sqrt(self._SquareAvg[i]) + self._Eps)
