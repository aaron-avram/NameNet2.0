"""
File containing the recurrent block
"""

import numpy as np
from net.cell import Cell
from net.block import Block
from net.tensor import Tensor

class Recurrent(Block):
    """
    Recurrent network block
    """
    cell: Cell

    def __init__(self, cell: Cell):
        super().__init__()
        self.cell = cell

    def forward(self, inp: Tensor):
        if len(inp.shape) == 2:
            inp = inp.reshape((1, inp.shape[0], inp.shape[1]))

        batch_size, seq_len, _ = inp.shape
        h = Tensor(np.zeros((batch_size, self.cell.hidden_size)))

        for t in range(seq_len):
            xt = inp[:, t, :]
            h = self.cell.step(h, xt)
        self.out = h
        return self.out

    def parameters(self):
        return self.cell.parameters()
