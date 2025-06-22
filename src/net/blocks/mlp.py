"""
File containing the MLP block class
"""

import numpy as np
from net.block import Block
from net.tensor import Tensor
from net.blocks.activation import Activation
from net.blocks.linear import Linear

class MLP(Block):
    """
    An MLP block with softmax probability and cross entropy loss
    """
    blocks: list[Block]

    def __init__(self, size: tuple, act: str, training: bool=True):
        super().__init__(training)
        self.blocks = []
        for l1, l2 in zip(size, size[1:]):
            self.blocks.append(Linear(l1, l2, training))
            self.blocks.append(Activation(act, training))
        self.blocks.pop()

    def forward(self, inp: Tensor) -> Tensor:
        self.inp = inp
        self.out = inp
        for block in self.blocks:
            self.out = block.forward(self.out)
        return self.out

    def _preds(self) -> Tensor:
        """
        Get predictions for each class -- For testing only
        """
        return np.argmax(self.out, axis=1)

    def _loss(self, targets: Tensor) -> Tensor:
        """
        Get the loss on the most recent forward pass -- For testing only
        """
        return self.out.cross_entropy(targets)

    def parameters(self) -> list[Tensor]:
        out = []
        for block in self.blocks:
            out.extend(block.parameters())
        return out
