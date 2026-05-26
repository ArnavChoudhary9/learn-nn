"""Cross-entropy loss (expects probabilities and one-hot targets, shape (C, B))."""

import numpy as np

from ..core.tensor import Tensor
from ..autograd.ops.add import Add
from ..autograd.ops.mul import Mul
from ..autograd.ops.neg import Neg
from ..autograd.ops.log import Log
from ..autograd.ops.sum import Sum
from ..autograd.ops.div import Div


_EPS = 1e-7


class CELoss:
    """L = -(1/B) * sum_{b,c} y_{c,b} * log(p_{c,b})

    Equivalent to: mean over batch of (sum over classes of -y * log p).
    """

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.Forward(y_pred, y_true)

    def Forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps_t = Tensor(np.full(y_pred.Shape, _EPS, dtype=np.float32))
        p_safe = Add.apply(y_pred, eps_t)
        log_p = Log.apply(p_safe)
        product = Mul.apply(y_true, log_p)

        batch_size = y_true.Shape[-1] if y_true.Data.ndim > 1 else 1
        return Neg.apply(Div.apply(Sum.apply(product), float(batch_size)))
