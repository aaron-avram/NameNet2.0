"""
Run tests for Network class
--- Written by Chat GPT ---
"""

import numpy as np
from net.tensor import Tensor
from net.cells.vanilla import Vanilla
from net.blocks.mlp import MLP
from net.blocks.recurrent import Recurrent
from net.network import Network
from net.train import sgd

def test_toy_sequence_classification():
    # Set seed for reproducibility
    np.random.seed(42)

    # --- Toy Data ---
    X_raw = np.array([
        [[1, 0], [1, 0], [0, 1]],  # class 0
        [[0, 1], [0, 1], [1, 0]],  # class 1
        [[1, 0], [1, 0], [1, 0]],  # class 0
        [[0, 1], [0, 1], [0, 1]],  # class 1
    ])
    Y_raw = np.array([0, 1, 0, 1])

    X = Tensor(X_raw)
    Y = Tensor(Y_raw)

    # --- Build Model ---
    cell = Vanilla(hidden_size=4, inp_size=2)
    recurrent = Recurrent(cell)
    mlp = MLP((4, 2), act="tanh")
    model = Network((recurrent, mlp))

    # --- Train ---
    sgd(model, X, Y, batch_size=len(X))

    # --- Evaluate ---
    model.training = False
    preds = [model.forward(x_seq).item() for x_seq in X]
    correct = sum(int(p == y) for p, y in zip(preds, Y_raw))

    # Assert full accuracy (perfect fit on toy data)
    assert correct == len(Y_raw), f"Expected all correct, got {correct}/{len(Y_raw)}"
