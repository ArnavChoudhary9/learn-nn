"""Computation graph node."""


class Node:
    """Represents one op in the autograd graph."""

    def __init__(self, fn_class, ctx, inputs):
        self.fn_class = fn_class
        self.ctx = ctx
        self.inputs = inputs  # original apply() args (Tensors or scalars)

    def __call__(self, grad_output):
        grads = self.fn_class.Backward(self.ctx, grad_output)
        if not isinstance(grads, tuple):
            grads = (grads,)
        return grads


class AutoGradGraph:
    def __init__(self):
        self.nodes = []
