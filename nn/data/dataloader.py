"""DataLoader."""

from typing import Iterator
import numpy as np
from nn.data.dataset import Dataset


class DataLoader:
    """Iterates over a Dataset in mini-batches.

    Yields tuples of Tensors (matching Dataset.__getitem__ structure) with
    shape (features, batchSize) per tensor, consistent with the project's
    (features, batch) convention.
    """

    def __init__(self, dataset: Dataset, batchSize: int, shuffle: bool = False) -> None:
        if batchSize < 1:
            raise ValueError(f"batchSize must be >= 1, got {batchSize}.")
        self._dataset = dataset
        self._batchSize = batchSize
        self._shuffle = shuffle

    @property
    def Dataset(self) -> Dataset:
        return self._dataset

    @property
    def BatchSize(self) -> int:
        return self._batchSize

    @property
    def Shuffle(self) -> bool:
        return self._shuffle

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self._dataset) + self._batchSize - 1) // self._batchSize

    def __iter__(self) -> Iterator:
        indices = np.arange(len(self._dataset))
        if self._shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(self._dataset), self._batchSize):
            batch_indices = indices[start : start + self._batchSize]
            yield self._dataset[batch_indices]
