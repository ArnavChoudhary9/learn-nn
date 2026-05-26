"""Backward engine — topological sort + gradient accumulation."""

import numpy as np
from ..core.tensor import Tensor


def backward(root: Tensor, grad: np.ndarray | None = None) -> None:
    """Run reverse-mode autodiff from `root` tensor."""
    if grad is None:
        grad = np.ones(root.Shape, dtype=np.float32)

    # Build topological order (inputs before outputs)
    order: list[Tensor] = []
    visited: set[int] = set()

    def _topo(t: Tensor) -> None:
        if id(t) in visited:
            return
        visited.add(id(t))
        if t._GradFn is not None:
            for inp in t._GradFn.inputs:
                if isinstance(inp, Tensor):
                    _topo(inp)
        order.append(t)

    _topo(root)

    root._Grad = grad

    for t in reversed(order):
        if t._GradFn is None or t._Grad is None:
            continue
        grad_tensor = Tensor(t._Grad)
        grads = t._GradFn(grad_tensor)
        for inp, g in zip(t._GradFn.inputs, grads):
            if isinstance(inp, Tensor) and inp.RequiresGrad and g is not None:
                if inp._Grad is None:
                    inp._Grad = np.zeros(inp.Shape, dtype=np.float32)
                inp._Grad = inp._Grad + g.Data
