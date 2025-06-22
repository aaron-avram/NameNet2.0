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

    def __init__(self, hidden_size: int = None, input_size: int = None, act: str = 'tanh', training: bool = True):
        self.initial_state = Tensor(np.zeros(shape=(hidden_size,)))
        self.training = training
        self.activation = Activation(act, training)

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
