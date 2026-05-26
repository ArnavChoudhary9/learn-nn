"""Stochastic Gradient Descent."""

from ..core.parameter import Parameter
from .base import Optimizer


class SGD(Optimizer):
    """theta <- theta - lr * grad"""

    def __init__(self, parameters: list[Parameter], lr: float = 0.01):
        super().__init__(parameters, lr)

    def Step(self) -> None:
        for p in self._Parameters:
            if p.Grad is None:
                continue
            p.Data = p.Data - self._LearningRate * p.Grad
