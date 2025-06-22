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

    def __init__(self, cell: Cell, training: bool = True):
        super().__init__(training)
        self.cell = cell

    def forward(self, inp: Tensor):
        if len(inp.shape) == 2:
            inp = inp.reshape((1, inp.shape[0], inp.shape[1]))

        batch_size, seq_len, _ = inp.shape
        cur = Tensor(np.repeat(self.cell.initial_state.value[None, :], batch_size, axis=0))

        for t in range(seq_len):
            xt = inp[:, t, :]
            cur = self.cell.step(cur, xt)
        self.out = cur
        return self.out

    def parameters(self):
        return self.cell.parameters()
