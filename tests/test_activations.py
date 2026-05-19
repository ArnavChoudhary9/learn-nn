# tests/test_activations.py

import numpy as np
import pytest

from nn.core.tensor import Tensor

from nn.activations.sigmoid import Sigmoid
from nn.activations.relu import ReLU
from nn.activations.tanh import Tanh
from nn.activations.softmax import Softmax


# ============================================================
# HELPERS
# ============================================================

EPSILON = 1e-5


def numerical_gradient(
    activation,
    x_data,
    epsilon=EPSILON
):
    """
    Numerical gradient for:
        L = sum(activation(x))
    """

    numerical = np.zeros_like(x_data)

    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):

            original = x_data[i, j]

            # f(x + epsilon)
            x_data[i, j] = original + epsilon

            plus = np.sum(
                activation.Forward(
                    Tensor(x_data)
                ).Data
            )

            # f(x - epsilon)
            x_data[i, j] = original - epsilon

            minus = np.sum(
                activation.Forward(
                    Tensor(x_data)
                ).Data
            )

            # Restore
            x_data[i, j] = original

            numerical[i, j] = (
                (plus - minus)
                / (2 * epsilon)
            )

    return numerical


# ============================================================
# SIGMOID TESTS
# ============================================================

def test_sigmoid_forward_shape():
    activation = Sigmoid()
    X = Tensor(np.random.randn(4, 8))
    Y = activation.Forward(X)
    assert Y.Shape == X.Shape


def test_sigmoid_output_range():
    activation = Sigmoid()
    X = Tensor(np.random.randn(16, 32))
    Y = activation.Forward(X)

    assert np.all(Y.Data >= 0.0)
    assert np.all(Y.Data <= 1.0)


def test_sigmoid_gradient_check():
    activation = Sigmoid()
    X_data = np.random.randn(3, 4)
    X = Tensor(X_data.copy())
    Y = activation.Forward(X)

    dA = Tensor(np.ones_like(Y.Data))

    analytical = (
        activation.Backward(dA)
        .Data
        .copy()
    )

    numerical = numerical_gradient(
        activation,
        X_data.copy()
    )

    difference = np.linalg.norm(
        analytical - numerical
    )

    assert difference < 1e-6


# ============================================================
# TANH TESTS
# ============================================================

def test_tanh_forward_shape():
    activation = Tanh()
    X = Tensor(np.random.randn(5, 7))
    Y = activation.Forward(X)

    assert Y.Shape == X.Shape


def test_tanh_output_range():
    activation = Tanh()
    X = Tensor(np.random.randn(16, 16))
    Y = activation.Forward(X)

    assert np.all(Y.Data >= -1.0)
    assert np.all(Y.Data <= 1.0)


def test_tanh_zero():
    activation = Tanh()
    X = Tensor(np.zeros((2, 2)))
    Y = activation.Forward(X)

    assert np.allclose(Y.Data, 0.0)


def test_tanh_gradient_check():
    activation = Tanh()
    X_data = np.random.randn(4, 4)
    X = Tensor(X_data.copy())
    Y = activation.Forward(X)

    dA = Tensor(np.ones_like(Y.Data))

    analytical = (
        activation.Backward(dA)
        .Data
        .copy()
    )

    numerical = numerical_gradient(
        activation,
        X_data.copy()
    )

    difference = np.linalg.norm(
        analytical - numerical
    )

    assert difference < 1e-6


# ============================================================
# RELU TESTS
# ============================================================

def test_relu_forward_shape():
    activation = ReLU()
    X = Tensor(np.random.randn(3, 9))
    Y = activation.Forward(X)

    assert Y.Shape == X.Shape


def test_relu_negative_zeroing():
    activation = ReLU()

    X = Tensor(np.array([
        [-1.0, 0.0, 1.0]
    ]))

    Y = activation.Forward(X)

    expected = np.array([
        [0.0, 0.0, 1.0]
    ])

    assert np.array_equal(
        Y.Data,
        expected
    )


def test_relu_gradient_behavior():
    activation = ReLU()

    X = Tensor(np.array([
        [-1.0, 0.0, 2.0]
    ]))

    activation.Forward(X)

    dA = Tensor(np.ones((1, 3)))
    dX = activation.Backward(dA)

    expected = np.array([
        [0.0, 0.0, 1.0]
    ])

    assert np.array_equal(
        dX.Data,
        expected
    )


def test_relu_gradient_check():
    activation = ReLU()

    # Avoid exact zero
    X_data = (
        np.random.randn(4, 4)
        + 0.1
    )

    X = Tensor(X_data.copy())
    Y = activation.Forward(X)
    dA = Tensor(np.ones_like(Y.Data))

    analytical = (
        activation.Backward(dA)
        .Data
        .copy()
    )

    numerical = numerical_gradient(
        activation,
        X_data.copy()
    )

    difference = np.linalg.norm(
        analytical - numerical
    )

    assert difference < 1e-4


# ============================================================
# SOFTMAX TESTS
# ============================================================

def test_softmax_forward_shape():
    activation = Softmax()
    X = Tensor(np.random.randn(6, 10))
    Y = activation.Forward(X)

    assert Y.Shape == X.Shape


def test_softmax_probability_distribution():
    activation = Softmax()
    X = Tensor(np.random.randn(5, 8))
    Y = activation.Forward(X)

    column_sums = np.sum(
        Y.Data,
        axis=0
    )

    assert np.allclose(
        column_sums,
        1.0,
        atol=1e-6
    )


def test_softmax_output_range():
    activation = Softmax()
    X = Tensor(np.random.randn(4, 7))
    Y = activation.Forward(X)

    assert np.all(Y.Data >= 0.0)
    assert np.all(Y.Data <= 1.0)


def test_softmax_numerical_stability():
    activation = Softmax()

    X = Tensor(np.array([
        [1000.0, -1000.0],
        [1001.0, -999.0]
    ]))

    Y = activation.Forward(X)

    assert not np.isnan(Y.Data).any()
    assert not np.isinf(Y.Data).any()


def test_softmax_identical_values():
    activation = Softmax()
    X = Tensor(np.ones((4, 3)))
    Y = activation.Forward(X)

    expected = np.full(
        (4, 3),
        0.25
    )

    assert np.allclose(
        Y.Data,
        expected,
        atol=1e-6
    )


# ============================================================
# COMMON ACTIVATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "activation_class",
    [
        Sigmoid,
        ReLU,
        Tanh,
        Softmax
    ]
)
def test_activation_backward_without_forward(
    activation_class
):
    activation = activation_class()

    dA = Tensor(
        np.random.randn(3, 3)
    )

    with pytest.raises(ValueError):
        activation.Backward(dA)


@pytest.mark.parametrize(
    "activation_class",
    [
        Sigmoid,
        ReLU,
        Tanh,
        Softmax
    ]
)
def test_activation_backward_shape(
    activation_class
):
    activation = activation_class()

    X = Tensor(
        np.random.randn(8, 5)
    )

    Y = activation.Forward(X)

    dA = Tensor(
        np.random.randn(*Y.Shape)
    )

    dX = activation.Backward(dA)

    assert dX.Shape == X.Shape


@pytest.mark.parametrize(
    "activation_class",
    [
        Sigmoid,
        ReLU,
        Tanh,
        Softmax
    ]
)
def test_activation_no_nan_or_inf(
    activation_class
):
    activation = activation_class()

    X = Tensor(
        np.random.randn(16, 16) * 100
    )

    Y = activation.Forward(X)

    assert not np.isnan(Y.Data).any()
    assert not np.isinf(Y.Data).any()
