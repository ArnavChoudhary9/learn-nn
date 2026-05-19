"""Mean squared error loss."""

from ..core.tensor import Tensor


class MSELoss:
    """Mean squared error loss."""

    _Prediction : Tensor | None
    _Target     : Tensor | None

    def __init__(self):
        """Initialize the mean squared error loss."""
        self._Prediction = None
        self._Target     = None

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        """Calculate the mean squared error loss."""
        return self.Forward(y_pred, y_true)
    
    def Forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Calculate the mean squared error loss."""

        self._Prediction = y_pred
        self._Target     = y_true

        y = y_true.Data
        y1 = y_pred.Data

        return Tensor(((y1 - y) ** 2).mean())
    
    def Backward(self) -> Tensor:
        """Calculate the gradient of the mean squared error loss."""
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
        return Tensor((2 / y1.size) * (y1 - y))
