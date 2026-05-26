"""Autograd context — stores values needed by the backward pass."""

from typing import Any


class Context:
    _Saved: tuple[Any, ...]

    def __init__(self):
        self._Saved: tuple[Any, ...] = ()

    def SaveForBackward(self, *args: Any):
        self._Saved = args

    @property
    def SavedTensors(self) -> tuple[Any, ...]:
        return self._Saved
