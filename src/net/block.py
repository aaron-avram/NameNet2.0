"""
File containing the abstract block class
"""

from net.tensor import Tensor

class Block:
    """
    A class representing a block in a neural network
    """

    inp: Tensor
    out: Tensor

    def __init__(self):
        self.inp = None
        self.out = None

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass in the abstract block class
        """
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """
        Return the parameters of the block
        """
        raise NotImplementedError
