"""
Run tests for tensor class
--- Written by Chat GPT ---
"""

import numpy as np
from net.tensor import Tensor

def test_scalar_addition():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    assert np.allclose(c.value, 5.0)
    c.backward()
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 1.0)

def test_tensor_addition_and_broadcasting():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[1.0], [2.0]]))  # will broadcast to (2, 2)
    c = a + b
    assert c.value.shape == (2, 2)
    c.backward(np.ones_like(c.value))
    assert np.allclose(a.grad, np.ones_like(a.value))
    assert np.allclose(b.grad, np.array([[2.0], [2.0]]))  # summed over columns

def test_matmul_backward():
    a = Tensor(np.random.randn(3, 4))
    b = Tensor(np.random.randn(4, 2))
    c = a @ b
    upstream = np.ones_like(c.value)
    c.backward(upstream)
    assert np.allclose(a.grad, upstream @ b.value.T)
    assert np.allclose(b.grad, a.value.T @ upstream)

def test_mul_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[2.0, 0.5], [1.0, 3.0]]))
    c = a * b
    c.backward(np.ones_like(c.value))
    assert np.allclose(a.grad, b.value)
    assert np.allclose(b.grad, a.value)

def test_tanh_backward():
    a = Tensor(np.array([[0.0, 1.0], [-1.0, 0.5]]))
    b = a.tanh()
    b.backward(np.ones_like(b.value))
    expected_grad = 1 - np.tanh(a.value) ** 2
    assert np.allclose(a.grad, expected_grad)

def test_relu_backward():
    a = Tensor(np.array([[-1.0, 0.0], [1.0, 2.0]]))
    b = a.relu()
    b.backward(np.ones_like(b.value))
    expected_grad = (a.value > 0).astype(float)
    assert np.allclose(a.grad, expected_grad)

def test_transpose_backward():
    a = Tensor(np.random.randn(2, 3))
    b = a.T
    c = b * 2.0
    c.backward(np.ones_like(c.value))
    assert a.grad.shape == a.value.shape
    assert np.allclose(a.grad, np.full_like(a.value, 2.0))

def test_cross_entropy():
    logits = Tensor(np.array([[2.0, 1.0, 0.1]]), grad_required=True)
    targets = Tensor(np.array([0]))
    loss = logits.cross_entropy(targets)
    loss.backward()
    assert loss.value.shape == ()
    assert logits.grad.shape == logits.value.shape

def test_array_ufunc_compatibility():
    a = Tensor(np.array([0.0, np.pi / 2]))
    b = np.sin(a)
    expected = np.sin(a.value)
    assert isinstance(b, Tensor)
    assert np.allclose(b.value, expected)

def test_item_scalar_extraction():
    scalar = Tensor(np.array(3.14))
    assert np.isclose(scalar.item(), 3.14)

def test_zero_grad():
    a = Tensor(np.array([1.0, 2.0]), grad_required=True)
    b = a * 2
    b.backward(np.ones_like(b.value))
    a.zero_grad_deep()
    assert np.allclose(a.grad, np.zeros_like(a.value))
