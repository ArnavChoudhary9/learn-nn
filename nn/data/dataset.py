"""Dataset abstraction."""

from abc import abstractmethod
import numpy as np
from nn.core.tensor import Tensor


class Dataset:
    """Abstract base class for all datasets."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping one or more Tensors.

    Each Tensor must have the same number of samples in the last axis.
    Indexing returns a tuple of single-sample Tensors with shape (features, 1).
    """

    def __init__(self, *tensors: Tensor) -> None:
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor.")
        sizes = [t.Shape[-1] for t in tensors]
        if len(set(sizes)) > 1:
            raise ValueError(
                f"All tensors must have the same number of samples in the last axis, got {sizes}."
            )
        self._tensors = tensors

    @property
    def Tensors(self) -> tuple:
        return self._tensors

    def __len__(self) -> int:
        return self._tensors[0].Shape[-1]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")
            return tuple(Tensor(t.Data[..., index : index + 1]) for t in self._tensors)
        # Array/slice batch indexing — avoids per-sample Python loop in DataLoader
        return tuple(Tensor(t.Data[..., index]) for t in self._tensors)
