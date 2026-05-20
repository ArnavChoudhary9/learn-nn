"""SGD optimizer."""

from ..core.parameter import Parameter

import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer."""

    _LearningRate: float
    _Parameters: list[Parameter]

    def __init__(self, parameters: list[Parameter], lr: float = 0.01):
        self._LearningRate = lr
        self._Parameters   = parameters

    def Step(self):
        """Update parameters using their gradients."""
        for parameter in self._Parameters:
            parameter.Data -= (
                self._LearningRate * parameter.Grad
                if parameter.Grad is not None else 0
            )
    
    def ZeroGrad(self):
        """Reset gradients of all parameters to zero."""
        for parameter in self._Parameters:
            if parameter.Grad is not None:
                parameter.Grad.fill(0.0)
            else:
                parameter.Grad = np.zeros(parameter.Data.shape, dtype=np.float32)
