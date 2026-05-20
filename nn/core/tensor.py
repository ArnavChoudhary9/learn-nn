"""Tensor primitive."""

import numpy as np

class Tensor:
    """Tensor primitive."""

    _Data: np.ndarray
    _Grad: np.ndarray | None

    _RequiresGrad: bool = False

    def __init__(self, data: np.ndarray | list[list[float]], requiresGrad: bool = False):
        self._Data = np.asarray(data, dtype=np.float32)
        self._RequiresGrad = requiresGrad

        self._Grad = (
            np.zeros(self._Data.shape, dtype=np.float32)
            if requiresGrad
            else None
        )

    @property
    def Data(self) -> np.ndarray:
        """Data."""
        return self._Data
    
    @Data.setter
    def Data(self, value: np.ndarray):
        """Set data."""
        self._Data = value

    @property
    def Grad(self) -> np.ndarray | None:
        """Gradient."""
        return self._Grad

    @Grad.setter
    def Grad(self, value: np.ndarray | None):
        """Set gradient."""
        self._Grad = value

    @property
    def RequiresGrad(self) -> bool:
        """Whether gradient is required."""
        return self._RequiresGrad
    
    @property
    def Shape(self) -> tuple[int, ...]:
        """Shape."""
        return self._Data.shape
