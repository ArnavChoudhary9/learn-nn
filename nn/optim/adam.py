"""Adam optimizer."""

import numpy as np

from ..core.parameter import Parameter
from .base import Optimizer


class Adam(Optimizer):
    """Adam: adaptive moments with bias-corrected first and second moments."""

    _Beta1: float
    _Beta2: float
    _Eps: float
    _Step: int
    _M: list[np.ndarray]
    _V: list[np.ndarray]

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(parameters, lr)
        self._Beta1 = beta1
        self._Beta2 = beta2
        self._Eps = eps
        self._Step = 0
        self._M = [np.zeros_like(p.Data, dtype=np.float32) for p in parameters]
        self._V = [np.zeros_like(p.Data, dtype=np.float32) for p in parameters]

    def Step(self) -> None:
        self._Step += 1
        b1, b2 = self._Beta1, self._Beta2
        bc1 = 1.0 - b1 ** self._Step
        bc2 = 1.0 - b2 ** self._Step

        for i, p in enumerate(self._Parameters):
            if p.Grad is None:
                continue
            g = p.Grad
            self._M[i] = b1 * self._M[i] + (1.0 - b1) * g
            self._V[i] = b2 * self._V[i] + (1.0 - b2) * g * g
            m_hat = self._M[i] / bc1
            v_hat = self._V[i] / bc2
            p.Data = p.Data - self._LearningRate * m_hat / (np.sqrt(v_hat) + self._Eps)
