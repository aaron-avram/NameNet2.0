"""
File containing Embedding block class
"""

import numpy as np
from net.tensor import Tensor
from net.block import Block

class Embedding(Block):
    """
    Class supporting an embedding block in a neural network
    """

    emb_matrix: Tensor

    def __init__(self, emb_size: int, vocab_size: int):
        super().__init__()
        self.emb_matrix = Tensor(np.random.randn(vocab_size, emb_size))

    def forward(self, inp):
        self.inp = inp
        self.out = inp.embed(self.emb_matrix)
        return self.out

    def parameters(self):
        return [self.emb_matrix]
