"""
File containing the linear block class
"""

import numpy as np
from net.block import Block
from net.tensor import Tensor

class Linear(Block):
    """
    Class supporting linear block functionality
    """

    weights: Tensor
    bias: Tensor

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        self.weights = Tensor(np.random.randn(out_size, inp_size)* np.sqrt(2.0 / inp_size))
        self.bias = Tensor(np.zeros(shape=(out_size,)))

    def forward(self, inp: Tensor) -> Tensor:
        self.inp = inp
        self.out = self.inp @ self.weights.T + self.bias

        return self.out

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.bias]
