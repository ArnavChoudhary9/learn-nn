"""Xavier/Glorot initialization."""

import numpy as np

def XavierInitialization(shape: tuple[int, ...]) -> np.ndarray:
    """Xavier/Glorot initialization."""
    if len(shape) < 2:
        raise ValueError("Xavier initialization requires at least 2 dimensions.")
    
    fanIn = shape[-2]
    fanOut = shape[-1]
    limit = np.sqrt(6 / (fanIn + fanOut))
    
    return np.random.uniform(-limit, limit, size=shape)
