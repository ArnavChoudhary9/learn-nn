"""Mean squared error loss."""

from ..core.tensor import Tensor
from ..autograd.ops.sub import Sub
from ..autograd.ops.pow import Pow
from ..autograd.ops.mean import Mean


class MSELoss:
    """L = mean((y_pred - y_true)^2)"""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.Forward(y_pred, y_true)

    def Forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = Sub.apply(y_pred, y_true)
        return Mean.apply(Pow.apply(diff, 2))
