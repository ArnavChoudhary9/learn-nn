"""Normal initialization."""

import numpy as np

def NormalInitialization(shape: tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """Normal initialization."""
    return np.random.normal(loc=mean, scale=std, size=shape)
