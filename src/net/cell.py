"""
File containing the cell class
"""

from net.tensor import Tensor

class Cell:
    """
    Class for a recurrent cell in a RNN
    """

    def __init__(self, training: bool = True):
        self.training = training

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
