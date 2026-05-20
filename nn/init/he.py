"""He initialization."""

import numpy as np

def HeInitialization(shape: tuple[int, ...]) -> np.ndarray:
    """He initialization."""
    if len(shape) < 2:
        raise ValueError("He initialization requires at least 2 dimensions.")
    
    fanIn = shape[-2]
    limit = np.sqrt(2 / fanIn)
    
    return np.random.uniform(-limit, limit, size=shape)
