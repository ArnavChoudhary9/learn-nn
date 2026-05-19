"""Binary cross-entropy loss."""

from ..core.tensor import Tensor

import numpy as np

class BCELoss:
    """Binary cross-entropy loss."""

    _Prediction : Tensor | None
    _Target     : Tensor | None

    def __init__(self):
        """Initialize the mean squared error loss."""
        self._Prediction = None
        self._Target     = None

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculate the binary cross-entropy loss."""
        return self.Forward(y_pred, y_true)

    def Forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculate the binary cross-entropy loss.

        Args:
            y_pred: Predicted probabilities (values between 0 and 1).
            y_true: True labels (0 or 1).

        Returns:
            The binary cross-entropy loss.
        """
        # Avoid division by zero and log of zero
        epsilon = 1e-15
        y1 = np.clip(y_pred.Data, epsilon, 1 - epsilon)
        y = y_true.Data

        self._Prediction = y_pred
        self._Target     = y_true

        # Calculate binary cross-entropy loss
        loss = - (y * np.log(y1) + (1 - y) * np.log(1 - y1))
        return Tensor(loss)
    
    def Backward(self) -> Tensor:
        """Calculate the gradient of the binary cross-entropy loss."""
        if (
            self._Prediction is None
            or
            self._Target is None
        ):
            raise ValueError(
                "No forward pass before backward."
            )

        y = self._Target.Data
        y1 = self._Prediction.Data

        # Avoid division by zero
        epsilon = 1e-15
        y1 = np.clip(y1, epsilon, 1 - epsilon)

        # Calculate the gradient of the binary cross-entropy loss
        grad = (y1 - y) / (y1 * (1 - y1))
        return Tensor(grad)
