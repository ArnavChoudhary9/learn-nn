"""Binary cross-entropy loss."""

import numpy as np

from ..core.tensor import Tensor
from ..autograd.ops.add import Add
from ..autograd.ops.sub import Sub
from ..autograd.ops.mul import Mul
from ..autograd.ops.neg import Neg
from ..autograd.ops.log import Log
from ..autograd.ops.mean import Mean


_EPS = 1e-7


class BCELoss:
    """L = -mean( y*log(p) + (1-y)*log(1-p) )"""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.Forward(y_pred, y_true)

    def Forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps_t = Tensor(np.full(y_pred.Shape, _EPS, dtype=np.float32))
        ones = Tensor(np.ones(y_pred.Shape, dtype=np.float32))
        ones_y = Tensor(np.ones(y_true.Shape, dtype=np.float32))

        p_safe = Add.apply(y_pred, eps_t)
        one_minus_p = Add.apply(Sub.apply(ones, y_pred), eps_t)
        one_minus_y = Sub.apply(ones_y, y_true)

        term1 = Mul.apply(y_true, Log.apply(p_safe))
        term2 = Mul.apply(one_minus_y, Log.apply(one_minus_p))

        return Neg.apply(Mean.apply(Add.apply(term1, term2)))
