"""Shared gradient utilities for ops."""

import numpy as np


def UnbroadcastGrad(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce a gradient array to match the original tensor shape.

    Handles the case where `a + b` broadcast b's shape during forward —
    we must sum over those broadcast dimensions going backward.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
