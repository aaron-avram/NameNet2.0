"""
File containing the vanilla RNN cell
"""

import numpy as np
from net.tensor import Tensor
from net.cell import Cell

class Vanilla(Cell):
    """
    Vanilla RNN cell
    """
    h_weight: Tensor
    x_weight: Tensor
    bias: Tensor

    def __init__(self, hidden_size: int, inp_size: int, act: str='tanh', training: bool = True):
        super().__init__(hidden_size=hidden_size, input_size=inp_size, act=act, training=training)
        self.h_weight = Tensor(np.random.rand(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.x_weight = Tensor(np.random.rand(hidden_size, inp_size) * np.sqrt(2.0 / inp_size))
        self.bias = Tensor(np.zeros(shape=(hidden_size,)))

    def step(self, prev: Tensor, inp: Tensor) -> Tensor:
        unact = prev @ self.h_weight + inp @ self.x_weight.T + self.bias
        return self.activation.forward(unact)

    def parameters(self):
        return [self.h_weight, self.x_weight, self.bias]
