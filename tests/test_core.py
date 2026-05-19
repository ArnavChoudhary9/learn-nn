# tests/test_core.py

import numpy as np

from nn.core.tensor import Tensor
from nn.core.parameter import Parameter
from nn.core.module import Module


# ============================================================
# DUMMY MODULE FOR TESTING
# ============================================================

class DummyModule(Module):
    def __init__(self):
        super().__init__()

        self.AddParameter(
            "weight",
            Parameter(np.random.randn(2, 2))
        )

    def Forward(self, x):
        return x
    
    def Backward(self, dZ):
        return dZ


class NestedModule(Module):
    def __init__(self):
        super().__init__()

        self.AddModule(
            "child1",
            DummyModule()
        )

        self.AddModule(
            "child2",
            DummyModule()
        )

    def Forward(self, x):
        return x
    
    def Backward(self, dZ):
        return dZ


# ============================================================
# TENSOR TESTS
# ============================================================

def test_tensor_shape():
    data = np.random.randn(3, 4)
    tensor = Tensor(data)
    assert tensor.Shape == (3, 4)


def test_tensor_requires_grad_default():
    tensor = Tensor(np.zeros((2, 2)))
    assert tensor.RequiresGrad is False


def test_tensor_grad_initialization():
    tensor = Tensor(
        np.zeros((3, 3)),
        requiresGrad=True
    )

    assert tensor.Grad.shape == (3, 3)


# ============================================================
# PARAMETER TESTS
# ============================================================

def test_parameter_requires_grad():
    parameter = Parameter(
        np.random.randn(2, 2)
    )

    assert parameter.RequiresGrad is True


# ============================================================
# MODULE TESTS
# ============================================================

def test_module_parameter_registration():
    module = DummyModule()
    params = module.Parameters

    assert len(params) == 1


def test_module_recursive_parameters():
    module = NestedModule()
    params = module.Parameters

    assert len(params) == 2


def test_module_child_registration():
    module = NestedModule()

    assert len(module._Modules) == 2
