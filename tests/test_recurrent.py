"""
Run tests for Reccurent class
--- Written by Chat GPT ---
"""

import numpy as np
from net.blocks.recurrent import Recurrent
from net.cells.vanilla import Vanilla
from net.tensor import Tensor

def test_recurrent_forward_and_backward():
    # Set up input sequence of 4 time steps with 3 input features
    seq_len = 4
    inp_size = 3
    hidden_size = 2
    xs = [Tensor(np.random.randn(inp_size), grad_required=True) for _ in range(seq_len)]

    # Set up recurrent block
    cell = Vanilla(hidden_size=hidden_size, inp_size=inp_size)
    rnn = Recurrent(cell=cell)

    # Forward
    out = rnn.forward(xs)
    assert out.shape == (hidden_size,), f"Output shape should be ({hidden_size},), got {out.shape}"

    # Dummy loss: sum of outputs
    loss = out.value.sum()
    loss_tensor = Tensor(loss, parents=(out,), op="sum", grad_required=True)
    loss_tensor.grad = np.ones_like(loss_tensor.value)
    loss_tensor.backward()

    # Check grads exist
    for i, x in enumerate(xs):
        assert x.grad is not None, f"Input at time step {i} has no gradient"

    for p in rnn.parameters():
        assert p.grad is not None, "Parameter has no gradient"
