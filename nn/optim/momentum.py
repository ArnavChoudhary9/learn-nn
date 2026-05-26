"""SGD with momentum."""

import numpy as np

from ..core.parameter import Parameter
from .base import Optimizer


class Momentum(Optimizer):
    """v <- mu * v + grad ; theta <- theta - lr * v"""

    _Mu: float
    _Velocity: list[np.ndarray]

    def __init__(self, parameters: list[Parameter], lr: float = 0.01, momentum: float = 0.9):
        super().__init__(parameters, lr)
        self._Mu = momentum
        self._Velocity = [np.zeros_like(p.Data, dtype=np.float32) for p in parameters]

    def Step(self) -> None:
        for i, p in enumerate(self._Parameters):
            if p.Grad is None:
                continue
            self._Velocity[i] = self._Mu * self._Velocity[i] + p.Grad
            p.Data = p.Data - self._LearningRate * self._Velocity[i]
