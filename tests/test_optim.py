# tests/test_optim.py

import numpy as np

from nn.core.parameter import Parameter
from nn.layers.linear import Linear
from nn.layers.sequential import Sequential
from nn.activations.relu import ReLU
from nn.optim.sgd import SGD

from nn.core.tensor import Tensor


# ============================================================
# TEST 1 — SINGLE PARAMETER UPDATE
# ============================================================

def test_sgd_single_parameter_update():

    parameter = Parameter(
        np.array([[1.0]])
    )

    parameter.Grad = np.array([[0.5]])

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    optimizer.Step()

    expected = np.array([[0.95]])

    assert np.allclose(
        parameter.Data,
        expected
    )


# ============================================================
# TEST 2 — ZERO GRAD
# ============================================================

def test_sgd_zero_grad():

    parameter = Parameter(
        np.random.randn(3, 4)
    )

    parameter.Grad = np.random.randn(3, 4)

    optimizer = SGD(
        [parameter],
        lr=0.01
    )

    optimizer.ZeroGrad()

    assert np.allclose(
        parameter.Grad,
        np.zeros((3, 4))
    )


# ============================================================
# TEST 3 — MULTIPLE PARAMETERS
# ============================================================

def test_sgd_multiple_parameters():

    p1 = Parameter(
        np.array([[1.0]])
    )

    p2 = Parameter(
        np.array([[2.0]])
    )

    p1.Grad = np.array([[0.1]])
    p2.Grad = np.array([[0.2]])

    optimizer = SGD(
        [p1, p2],
        lr=0.1
    )

    optimizer.Step()

    assert np.allclose(
        p1.Data,
        np.array([[0.99]])
    )

    assert np.allclose(
        p2.Data,
        np.array([[1.98]])
    )


# ============================================================
# TEST 4 — UPDATE DIRECTION
# ============================================================

def test_sgd_update_direction():

    parameter = Parameter(
        np.array([[1.0]])
    )

    parameter.Grad = np.array([[1.0]])

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    optimizer.Step()

    # Positive gradient
    # should reduce parameter

    assert parameter.Data[0, 0] < 1.0


# ============================================================
# TEST 5 — NEGATIVE GRADIENT
# ============================================================

def test_sgd_negative_gradient():

    parameter = Parameter(
        np.array([[1.0]])
    )

    parameter.Grad = np.array([[-1.0]])

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    optimizer.Step()

    # Negative gradient
    # should increase parameter

    assert parameter.Data[0, 0] > 1.0


# ============================================================
# TEST 6 — ZERO GRADIENT
# ============================================================

def test_sgd_zero_gradient():

    parameter = Parameter(
        np.array([[1.0]])
    )

    parameter.Grad = np.array([[0.0]])

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    before = parameter.Data.copy()

    optimizer.Step()

    assert np.allclose(
        parameter.Data,
        before
    )


# ============================================================
# TEST 7 — MODEL PARAMETER UPDATE
# ============================================================

def test_sgd_model_parameter_update():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    parameters = model.Parameters

    optimizer = SGD(
        parameters,
        lr=0.01
    )

    X = Tensor(
        np.random.randn(2, 8)
    )

    Y = model.Forward(X)

    dY = Tensor(
        np.random.randn(1, 8)
    )

    model.Backward(dY)

    before = [
        parameter.Data.copy()
        for parameter in parameters
    ]

    optimizer.Step()

    after = [
        parameter.Data
        for parameter in parameters
    ]

    changed = False

    for b, a in zip(before, after):

        if not np.allclose(b, a):
            changed = True

    assert changed


# ============================================================
# TEST 8 — ZERO GRAD AFTER BACKPROP
# ============================================================

def test_sgd_zero_grad_after_backprop():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    X = Tensor(
        np.random.randn(2, 8)
    )

    Y = model.Forward(X)

    dY = Tensor(
        np.random.randn(1, 8)
    )

    model.Backward(dY)

    optimizer = SGD(
        model.Parameters,
        lr=0.01
    )

    optimizer.ZeroGrad()

    for parameter in model.Parameters:

        assert np.allclose(
            parameter.Grad,
            np.zeros_like(parameter.Data)
        )


# ============================================================
# TEST 9 — MULTIPLE OPTIMIZER STEPS
# ============================================================

def test_multiple_optimizer_steps():

    parameter = Parameter(
        np.array([[1.0]])
    )

    parameter.Grad = np.array([[0.1]])

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    optimizer.Step()
    optimizer.Step()
    optimizer.Step()

    expected = 1.0 - 3 * 0.1 * 0.1

    assert np.isclose(
        parameter.Data[0, 0],
        expected
    )


# ============================================================
# TEST 10 — LARGE TENSOR UPDATE
# ============================================================

def test_large_tensor_update():

    parameter = Parameter(
        np.random.randn(128, 256)
    )

    parameter.Grad = np.random.randn(128, 256)

    before = parameter.Data.copy()

    optimizer = SGD(
        [parameter],
        lr=0.001
    )

    optimizer.Step()

    assert not np.allclose(
        before,
        parameter.Data
    )
