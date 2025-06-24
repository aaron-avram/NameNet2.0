"""
File containing the cell class
"""

import numpy as np
from net.tensor import Tensor
from net.blocks.activation import Activation

class Cell:
    """
    Class for a recurrent cell in a RNN
    """

    def __init__(self, hidden_size: int = None, input_size: int = None, act: str = 'tanh'):
        self.activation = Activation(act)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def step(self, prev: Tensor, inp: Tensor) -> Tensor:
        """
        Take a step with the cell and return the result
        """
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """
        Return the cell parameters
        """
        raise NotImplementedError
