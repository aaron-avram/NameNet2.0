"""
File containing the recurrent block
"""

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
        cur = self.cell.initial_state.broadcast_to((batch_size, self.cell.initial_state.shape[0]))

        for t in range(seq_len):
            xt = inp[:, t, :]
            cur = self.cell.step(cur, xt)
        self.out = cur
        return self.out

    def parameters(self):
        return self.cell.parameters()
