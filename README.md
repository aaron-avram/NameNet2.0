# NameNet2.0
NameNet but make it recurrent!

## Table of Contents
- [Plan](#plan)
- [Implementation](#implementation)
- [Takeaways](#takeaways)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Plan
In this project I will build up the mathematical formulations of a Vanilla Recurrent network
(See the writeups folder) then use this theory to build a character level language model for name generation.

## Implementation
I discuss the mathematical details of my training and overall model design
in my write up. I made this as a recurrent upgrade to my feedforward network for name generation. I also improved the NumPy code in the model architecture
designing my own personal automatic differentiation engine for the model.

## Takeaways
I was happy with the model, as it performed similarly to the feed forward version but had less bad names and more exotic/interesting names that made its generation feel a bit more natural.

## Acknowledgments

The ideas in **NameNet** are based on foundational research and inspiring implementations, including:

- [Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — *A Neural Probabilistic Language Model* provided the conceptual backbone for building a statistical language model using neural networks.
- [Andrej Karpathy's `char-rnn`](https://github.com/karpathy/char-rnn) — This repo inspired the character-level generation format and workflow.
- [Cybenko (1989)](https://www.sciencedirect.com/science/article/pii/0893608089900208) — *Approximation by superpositions of a sigmoidal function* laid the theoretical groundwork for universal function approximation with neural networks.

While all code and mathematical derivations were developed independently, these resources deeply informed the design and motivation of the project.

## License
This project is licensed under the [MIT License](LICENSE).
