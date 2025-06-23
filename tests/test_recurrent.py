"""
Run tests for Reccurent class
--- Written by Chat GPT ---
"""

import numpy as np
import pytest
from net.tensor import Tensor
from net.cells.vanilla import Vanilla
from net.blocks.recurrent import Recurrent


@pytest.fixture
def toy_input_batch():
    # Shape: (batch_size=2, seq_len=3, input_dim=2)
    np.random.seed(0)
    data = np.random.randn(2, 3, 2)
    return Tensor(data, grad_required=True)


def test_recurrent_output_shape(toy_input_batch):
    cell = Vanilla(hidden_size=4, inp_size=2)
    rnn = Recurrent(cell)
    
    output = rnn.forward(toy_input_batch)
    assert isinstance(output, Tensor)
    assert output.shape == (2, 4), f"Expected output shape (2, 4), got {output.shape}"


def test_recurrent_computational_graph(toy_input_batch):
    cell = Vanilla(hidden_size=4, inp_size=2)
    rnn = Recurrent(cell)

    out = rnn.forward(toy_input_batch)
    assert out.op is not None
    assert out.parents, "Output should have parents for backpropagation"


def test_recurrent_backward(toy_input_batch):
    cell = Vanilla(hidden_size=4, inp_size=2)
    rnn = Recurrent(cell)

    out = rnn.forward(toy_input_batch)
    loss = out.sum()  # scalar loss
    loss.backward()

    # Make sure input has gradient
    assert toy_input_batch.grad is not None
    assert toy_input_batch.grad.shape == toy_input_batch.value.shape


def test_recurrent_param_gradients(toy_input_batch):
    cell = Vanilla(hidden_size=4, inp_size=2)
    rnn = Recurrent(cell)

    out = rnn.forward(toy_input_batch)
    loss = out.sum()
    loss.backward()

    for param in rnn.parameters():
        assert param.grad is not None, f"Parameter {param} missing gradient"
        assert param.grad.shape == param.value.shape


def test_step_computation_independence():
    """
    Sanity check: step should behave consistently across time steps.
    """
    cell = Vanilla(hidden_size=4, inp_size=2)
    prev = Tensor(np.ones((2, 4)))
    xt = Tensor(np.ones((2, 2)))

    out1 = cell.step(prev, xt)
    out2 = cell.step(prev, xt)

    np.testing.assert_allclose(out1.value, out2.value, rtol=1e-5)
