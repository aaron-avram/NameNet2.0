"""
File containing the network class
"""

import numpy as np
from net.block import Block
from net.tensor import Tensor

class Network:
    """
    File containing a network class
    """

    blocks: list[Block]
    training: bool

    def __init__(self, blocks: tuple, training: bool = True):
        self.blocks = blocks
        self.training = training

    def forward(self, inp: Tensor, targets: Tensor = None) -> Tensor:
        """
        Forward pass through the network
        """
        cur_inp = inp
        for block in self.blocks:
            block.forward(cur_inp)

        if self.training:
            return self.loss(targets)
        return self.preds()

    def preds(self) -> Tensor:
        """
        Get predictions for each class
        """
        return np.argmax(self.blocks[-1].out, axis=1)

    def loss(self, targets: Tensor) -> Tensor:
        """
        Get the loss on the most recent forward pass
        """
        return self.blocks[-1].out.cross_entropy(targets)

    def parameters(self) -> list[Tensor]:
        """
        Get network parameters
        """
        out = []
        for block in self.blocks:
            out.extend(block.parameters())
