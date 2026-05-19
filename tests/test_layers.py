import numpy as np
import pytest

from nn.core.tensor import Tensor

from nn.layers.linear import Linear
from nn.layers.sequential import Sequential

from nn.activations.relu import ReLU
from nn.activations.sigmoid import Sigmoid


# ============================================================
# LINEAR LAYER TESTS
# ============================================================

def test_linear_forward_shape():

    batch_size = 5
    in_features = 2
    out_features = 4

    layer = Linear(
        in_features,
        out_features
    )

    X = Tensor(
        np.random.randn(
            in_features,
            batch_size
        )
    )

    Z = layer.Forward(X)

    assert Z.Shape == (
        out_features,
        batch_size
    )


def test_linear_backward_shapes():

    batch_size = 5
    in_features = 2
    out_features = 4

    layer = Linear(
        in_features,
        out_features
    )

    X = Tensor(
        np.random.randn(
            in_features,
            batch_size
        )
    )

    layer.Forward(X)

    dZ = Tensor(
        np.random.randn(
            out_features,
            batch_size
        )
    )

    dX = layer.Backward(dZ)

    dW = (
        layer._Parameters["weight"]
        .Grad
    )

    dB = (
        layer._Parameters["bias"]
        .Grad
    )

    assert dW.shape == (
        out_features,
        in_features
    )

    assert dB.shape == (
        out_features,
        1
    )

    assert dX.Shape == X.Shape


def test_linear_gradient_check():

    epsilon = 1e-5

    batch_size = 3
    in_features = 2
    out_features = 2

    layer = Linear(
        in_features,
        out_features
    )

    X = Tensor(
        np.random.randn(
            in_features,
            batch_size
        )
    )

    Z = layer.Forward(X)

    dZ = Tensor(
        np.ones_like(Z.Data)
    )

    layer.Backward(dZ)

    analytical_dW = (
        layer._Parameters["weight"]
        .Grad
        .copy()
    )

    numerical_dW = np.zeros_like(
        analytical_dW
    )

    W = (
        layer._Parameters["weight"]
        .Data
    )

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):

            original = W[i, j]

            # f(x + epsilon)
            W[i, j] = (
                original + epsilon
            )

            plus_loss = np.sum(
                layer.Forward(X).Data
            )

            # f(x - epsilon)
            W[i, j] = (
                original - epsilon
            )

            minus_loss = np.sum(
                layer.Forward(X).Data
            )

            # Restore
            W[i, j] = original

            numerical_dW[i, j] = (
                (plus_loss - minus_loss)
                / (2 * epsilon)
            )

    difference = np.linalg.norm(
        analytical_dW - numerical_dW
    )

    assert difference < 1e-6


def test_multiple_forward_backward_passes():

    layer = Linear(3, 2)

    for _ in range(10):

        X = Tensor(
            np.random.randn(3, 4)
        )

        layer.Forward(X)

        dZ = Tensor(
            np.random.randn(2, 4)
        )

        dX = layer.Backward(dZ)

        assert dX.Shape == X.Shape


def test_linear_no_nan_or_inf():

    layer = Linear(2, 2)

    X = Tensor(
        np.random.randn(2, 8) * 1000
    )

    Z = layer.Forward(X)

    assert not np.isnan(Z.Data).any()
    assert not np.isinf(Z.Data).any()

    dZ = Tensor(
        np.random.randn(2, 8)
    )

    dX = layer.Backward(dZ)

    assert not np.isnan(dX.Data).any()
    assert not np.isinf(dX.Data).any()


def test_linear_backward_without_forward():

    layer = Linear(2, 2)

    dZ = Tensor(
        np.random.randn(2, 2)
    )

    with pytest.raises(ValueError):
        layer.Backward(dZ)


def test_linear_invalid_input_shape():

    layer = Linear(2, 3)

    X = Tensor(
        np.random.randn(5, 4)
    )

    with pytest.raises(Exception):
        layer.Forward(X)


def test_linear_single_batch():

    layer = Linear(2, 4)

    X = Tensor(
        np.random.randn(2, 1)
    )

    Z = layer.Forward(X)

    assert Z.Shape == (4, 1)


def test_linear_large_batch():

    layer = Linear(16, 32)

    X = Tensor(
        np.random.randn(16, 512)
    )

    Z = layer.Forward(X)

    assert Z.Shape == (32, 512)


# ============================================================
# SEQUENTIAL TESTS
# ============================================================

def test_sequential_forward_shape():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    X = Tensor(
        np.random.randn(2, 8)
    )

    Y = model.Forward(X)

    assert Y.Shape == (1, 8)


def test_sequential_backward_shape():

    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 1)
    )

    X = Tensor(
        np.random.randn(2, 6)
    )

    model.Forward(X)

    dY = Tensor(
        np.random.randn(1, 6)
    )

    dX = model.Backward(dY)

    assert dX.Shape == X.Shape


def test_sequential_parameter_collection():

    model = Sequential(
        Linear(2, 4),
        Sigmoid(),
        Linear(4, 1)
    )

    parameters = model.Parameters

    assert len(parameters) == 4


def test_empty_sequential():

    model = Sequential()

    X = Tensor(
        np.random.randn(3, 5)
    )

    Y = model.Forward(X)

    assert Y.Shape == X.Shape


def test_sequential_backward_without_forward():

    model = Sequential(
        Linear(2, 2)
    )

    dY = Tensor(
        np.random.randn(2, 4)
    )

    with pytest.raises(ValueError):
        model.Backward(dY)


def test_deep_sequential_network():

    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 16),
        ReLU(),
        Linear(16, 8),
        Sigmoid(),
        Linear(8, 1)
    )

    X = Tensor(
        np.random.randn(2, 32)
    )

    Y = model.Forward(X)

    assert Y.Shape == (1, 32)


def test_repeated_sequential_forward_backward():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    for _ in range(10):

        X = Tensor(
            np.random.randn(2, 8)
        )

        model.Forward(X)

        dY = Tensor(
            np.random.randn(1, 8)
        )

        dX = model.Backward(dY)

        assert dX.Shape == X.Shape


def test_sequential_gradients_exist():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    X = Tensor(
        np.random.randn(2, 5)
    )

    model.Forward(X)

    dY = Tensor(
        np.random.randn(1, 5)
    )

    model.Backward(dY)

    parameters = model.Parameters

    for parameter in parameters:

        assert parameter.Grad is not None


def test_single_layer_sequential():

    model = Sequential(
        Linear(3, 7)
    )

    X = Tensor(
        np.random.randn(3, 4)
    )

    Y = model.Forward(X)

    assert Y.Shape == (7, 4)


def test_sequential_invalid_input_shape():

    model = Sequential(
        Linear(2, 4)
    )

    X = Tensor(
        np.random.randn(5, 3)
    )

    with pytest.raises(Exception):
        model.Forward(X)


def test_sequential_module_registration():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    assert len(model._Modules) == 3


def test_sequential_forward_consistency():

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )

    X = Tensor(
        np.random.randn(2, 5)
    )

    Y1 = model.Forward(X)
    Y2 = model.Forward(X)

    assert np.allclose(
        Y1.Data,
        Y2.Data
    )
