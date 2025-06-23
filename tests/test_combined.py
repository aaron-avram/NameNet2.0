"""
Run tests for Network class with embeddings
--- Written by Chat GPT ---
"""

import numpy as np
import pytest
from net.tensor import Tensor
from net.blocks.embedding import Embedding
from net.blocks.recurrent import Recurrent
from net.blocks.mlp import MLP
from net.cells.vanilla import Vanilla
from net.network import Network
from net.train import sgd

# Set random seed for reproducibility
np.random.seed(42)

# Toy dataset: encode sequences of token indices
X_raw = np.array([
    [0, 0, 1],  # class 0
    [1, 1, 0],  # class 1
    [0, 0, 0],  # class 0
    [1, 1, 1],  # class 1
])
Y_raw = np.array([0, 1, 0, 1])

X = Tensor(X_raw)
Y = Tensor(Y_raw)

# Define model
embedding = Embedding(emb_size=2, vocab_size=2)
cell = Vanilla(hidden_size=4, inp_size=2)
recurrent = Recurrent(cell)
mlp = MLP((4, 2), act="tanh")
model = Network((embedding, recurrent, mlp))

@pytest.mark.parametrize("epochs", [20])
def test_embedding_recurrent_mlp_learns_toy_sequence(epochs):
    sgd(model, X, Y, batch_size=4, steps=epochs)
    model.training = False
    correct = 0
    for x_seq, y_true in zip(X.value, Y.value):
        pred = model.forward(Tensor(np.array(x_seq, dtype=int)))
        if pred.item() == y_true:
            correct += 1
    assert correct >= 3, f"Model accuracy too low: {correct}/4"
