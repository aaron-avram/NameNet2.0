"""
File containing the activation block class
"""

from net.block import Block
from net.tensor import Tensor

class Activation(Block):
    """
    Activation layer in a neural network, only support Tanh or ReLU
    """

    act: str

    def __init__(self, act, training: bool):
        super().__init__(training)
        self.act = act

    def forward(self, inp: Tensor):
        self.inp = inp
        if self.act == 'tanh':
            self.out = inp.tanh()
        if self.act == 'relu':
            self.out = inp.relu()
        if self.out is not None:
            return self.out
        raise NotImplementedError

    def parameters(self):
        return []
