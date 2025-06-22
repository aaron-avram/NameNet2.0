"""
Run tests for MLP class
--- Written by Chat GPT ---
"""

import numpy as np
from net.blocks.mlp import MLP
from net.tensor import Tensor

def make_dummy_data(n=10, d_in=5, d_out=3, seed=42):
    np.random.seed(seed)
    X = Tensor(np.random.randn(n, d_in), grad_required=True)
    y = Tensor(np.random.randint(0, d_out, size=n), grad_required=False)
    return X, y

def test_forward_pass_output_shape():
    mlp = MLP(size=(5, 10, 3), act='relu')
    X, _ = make_dummy_data()
    out = mlp.forward(X)
    assert isinstance(out, Tensor)
    assert out.value.shape == (10, 3)

def test_predictions_are_valid():
    mlp = MLP(size=(5, 8, 3), act='tanh')
    X, _ = make_dummy_data()
    mlp.forward(X)
    preds = mlp._preds()
    assert preds.shape == (10,)
    assert np.all((preds >= 0) & (preds < 3))

def test_loss_computation():
    mlp = MLP(size=(5, 10, 3), act='tanh')
    X, y = make_dummy_data()
    mlp.forward(X)
    loss = mlp._loss(y)
    assert isinstance(loss, Tensor)
    assert loss.shape == ()
    assert loss.value > 0

def test_backward_shapes():
    mlp = MLP(size=(5, 6, 3), act='tanh')
    X, y = make_dummy_data()
    mlp.forward(X)
    loss = mlp._loss(y)
    loss.backward()

    # Ensure gradients are filled for all parameters
    for block in mlp.blocks:
        if hasattr(block, 'W'):
            assert block.W.grad.shape == block.W.value.shape
        if hasattr(block, 'b'):
            assert block.b.grad.shape == block.b.value.shape

def test_mlp_with_large_depth():
    mlp = MLP(size=(5, 20, 30, 20, 10, 3), act='relu')
    X, y = make_dummy_data()
    out = mlp.forward(X)
    assert out.value.shape == (10, 3)
    loss = mlp._loss(y)
    assert isinstance(loss, Tensor)

def test_zero_grad():
    mlp = MLP(size=(5, 6, 3), act='tanh')
    X, y = make_dummy_data()
    mlp.forward(X)
    loss = mlp._loss(y)
    loss.backward()

    # Zero gradients
    for block in mlp.blocks:
        if hasattr(block, 'W'):
            block.W.zero_grad()
            assert np.allclose(block.W.grad, 0)
        if hasattr(block, 'b'):
            block.b.zero_grad()
            assert np.allclose(block.b.grad, 0)
