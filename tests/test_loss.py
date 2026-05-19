# tests/test_loss.py

import numpy as np
import pytest

from nn.core.tensor import Tensor
from nn.losses.mse import MSELoss


# ============================================================
# TEST 1 — FORWARD RETURNS TENSOR
# ============================================================

def test_mse_forward_returns_tensor():

    loss = MSELoss()

    y_pred = Tensor(
        np.random.randn(1, 8)
    )

    y_true = Tensor(
        np.random.randn(1, 8)
    )

    output = loss.Forward(
        y_pred,
        y_true
    )

    assert isinstance(output, Tensor)


# ============================================================
# TEST 2 — FORWARD RETURNS SCALAR SHAPE
# ============================================================

def test_mse_forward_scalar_shape():

    loss = MSELoss()

    y_pred = Tensor(
        np.random.randn(2, 4)
    )

    y_true = Tensor(
        np.random.randn(2, 4)
    )

    output = loss.Forward(
        y_pred,
        y_true
    )

    assert output.Shape == ()


# ============================================================
# TEST 3 — ZERO LOSS
# ============================================================

def test_mse_zero_loss():

    loss = MSELoss()

    y_pred = Tensor(
        np.array([
            [1.0, 2.0, 3.0]
        ])
    )

    y_true = Tensor(
        np.array([
            [1.0, 2.0, 3.0]
        ])
    )

    output = loss.Forward(
        y_pred,
        y_true
    )

    assert np.isclose(
        output.Data,
        0.0
    )


# ============================================================
# TEST 4 — KNOWN LOSS VALUE
# ============================================================

def test_mse_known_value():

    loss = MSELoss()

    y_pred = Tensor(
        np.array([
            [2.0, 4.0]
        ])
    )

    y_true = Tensor(
        np.array([
            [1.0, 1.0]
        ])
    )

    output = loss.Forward(
        y_pred,
        y_true
    )

    # ((2-1)^2 + (4-1)^2) / 2
    # = (1 + 9) / 2
    # = 5

    assert np.isclose(
        output.Data,
        5.0
    )


# ============================================================
# TEST 5 — BACKWARD SHAPE
# ============================================================

def test_mse_backward_shape():

    loss = MSELoss()

    y_pred = Tensor(
        np.random.randn(4, 6)
    )

    y_true = Tensor(
        np.random.randn(4, 6)
    )

    loss.Forward(
        y_pred,
        y_true
    )

    dY = loss.Backward()

    assert dY.Shape == y_pred.Shape


# ============================================================
# TEST 6 — BACKWARD WITHOUT FORWARD
# ============================================================

def test_mse_backward_without_forward():

    loss = MSELoss()

    with pytest.raises(ValueError):
        loss.Backward()


# ============================================================
# TEST 7 — GRADIENT CHECK
# ============================================================

def test_mse_gradient_check():

    epsilon = 1e-5

    loss = MSELoss()

    y_pred_data = np.random.randn(3, 4)

    y_true_data = np.random.randn(3, 4)

    y_pred = Tensor(
        y_pred_data.copy()
    )

    y_true = Tensor(
        y_true_data.copy()
    )

    # Analytical gradient
    loss.Forward(
        y_pred,
        y_true
    )

    analytical = (
        loss.Backward()
        .Data
        .copy()
    )

    # Numerical gradient
    numerical = np.zeros_like(
        y_pred_data
    )

    for i in range(y_pred_data.shape[0]):
        for j in range(y_pred_data.shape[1]):

            original = y_pred_data[i, j]

            # f(x + epsilon)
            y_pred_data[i, j] = (
                original + epsilon
            )

            plus = (
                MSELoss().Forward(
                    Tensor(y_pred_data.copy()),
                    y_true
                ).Data
            )

            # f(x - epsilon)
            y_pred_data[i, j] = (
                original - epsilon
            )

            minus = (
                MSELoss().Forward(
                    Tensor(y_pred_data.copy()),
                    y_true
                ).Data
            )

            # Restore
            y_pred_data[i, j] = original

            numerical[i, j] = (
                (plus - minus)
                / (2 * epsilon)
            )

    difference = np.linalg.norm(
        analytical - numerical
    )

    assert difference < 1e-6


# ============================================================
# TEST 8 — NO NAN OR INF
# ============================================================

def test_mse_no_nan_or_inf():

    loss = MSELoss()

    y_pred = Tensor(
        np.random.randn(8, 8) * 1000
    )

    y_true = Tensor(
        np.random.randn(8, 8) * 1000
    )

    output = loss.Forward(
        y_pred,
        y_true
    )

    assert not np.isnan(output.Data)
    assert not np.isinf(output.Data)
    