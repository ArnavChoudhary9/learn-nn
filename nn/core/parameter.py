"""Trainable parameter."""

from .tensor import Tensor
import numpy as np

class Parameter(Tensor):
    """Trainable parameter."""

    def __init__(self, data: np.ndarray):
        super().__init__(data, requiresGrad=True)
    