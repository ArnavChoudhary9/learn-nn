"""Data transforms."""

from abc import abstractmethod
import numpy as np
from nn.core.tensor import Tensor


class Transform:
    """Abstract base class for all transforms."""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Compose(Transform):
    """Chain multiple transforms sequentially."""

    def __init__(self, transforms: list) -> None:
        if len(transforms) == 0:
            raise ValueError("Compose requires at least one transform.")
        self._transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        for t in self._transforms:
            x = t(x)
        return x


class Normalize(Transform):
    """Standardize a Tensor to zero mean and unit variance.

    Mean and std are computed per feature (across the last axis) on first call
    if not provided at construction time.
    """

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        self._mean = mean
        self._std = std

    @property
    def Mean(self) -> np.ndarray | None:
        return self._mean

    @property
    def Std(self) -> np.ndarray | None:
        return self._std

    def __call__(self, x: Tensor) -> Tensor:
        if self._mean is None:
            self._mean = x.Data.mean(axis=-1, keepdims=True)
        if self._std is None:
            self._std = x.Data.std(axis=-1, keepdims=True)

        std = np.where(self._std == 0, 1.0, self._std)
        return Tensor((x.Data - self._mean) / std)
