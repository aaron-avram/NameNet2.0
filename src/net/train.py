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

        l2_lambda = 1e-4
        l2_penalty = Tensor(0.0)
        for param in model.parameters():
            l2_penalty += (param * param).sum()

        loss = loss + l2_lambda * l2_penalty

        loss.zero_grad_deep()
        loss.backward()

        for param in model.parameters():
            param.clip_grad(5.0)
            param += -lr * param.grad
        

        
        if step == steps // 2:
            lr = lr * 0.1

        if step % 100 == 0:
            print(f"Loss: {loss.item()} on step: {step + 1}")
