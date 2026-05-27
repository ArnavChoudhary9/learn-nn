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

    By default the sample axis is the last axis of each tensor (the project's
    (features, batch) convention). Pass `batchAxis` to override — either an int
    applied to every tensor, or a tuple giving the batch axis per tensor. The
    latter is needed when mixing layouts in one batch (e.g. NCHW inputs paired
    with column-major targets for a CNN).
    """

    def __init__(self, *tensors: Tensor, batchAxis: int | tuple[int, ...] = -1) -> None:
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor.")
        if isinstance(batchAxis, int):
            axes = tuple(batchAxis for _ in tensors)
        else:
            axes = tuple(batchAxis)
            if len(axes) != len(tensors):
                raise ValueError(
                    f"batchAxis tuple length {len(axes)} does not match tensor count {len(tensors)}."
                )
        sizes = [t.Shape[ax] for t, ax in zip(tensors, axes)]
        if len(set(sizes)) > 1:
            raise ValueError(
                f"All tensors must have the same number of samples along their batch axis, got {sizes}."
            )
        self._tensors = tensors
        self._batchAxes = axes

    @property
    def Tensors(self) -> tuple:
        return self._tensors

    def __len__(self) -> int:
        return self._tensors[0].Shape[self._batchAxes[0]]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")
            sel = slice(index, index + 1)
        else:
            sel = index
        # Build a tuple indexer per tensor — direct advanced indexing is ~10000x
        # faster than np.take on the last axis of a 2D matrix in NumPy.
        return tuple(
            Tensor(t.Data[_AxisIndexer(t.Data.ndim, ax, sel)])
            for t, ax in zip(self._tensors, self._batchAxes)
        )


def _AxisIndexer(ndim: int, axis: int, sel) -> tuple:
    if axis < 0:
        axis += ndim
    return (slice(None),) * axis + (sel,) + (slice(None),) * (ndim - axis - 1)
