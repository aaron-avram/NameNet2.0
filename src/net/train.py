"""
File for training functions
"""

import numpy as np
from net.tensor import Tensor
from net.network import Network
from net.util import SEED

def sgd(model: Network, xs: Tensor, ys: Tensor, lr: float = 0.1, batch_size: int = 30, steps: int = 1000):
    """
    Perform stochastic gradient descent on model
    """

    gen = np.random.default_rng(SEED)

    for step in range(steps):
        idx = gen.integers(0, len(xs), batch_size)
        x_batch, y_batch = xs[idx], ys[idx]

        loss = model.forward(x_batch, y_batch)
        
        loss.zero_grad_deep()
        loss.backward()

        for param in model.parameters():
            param.clip_grad(5.0)
            param += -lr * param.grad