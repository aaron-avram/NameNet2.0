"""
Character level tokenizer
"""

import numpy as np

def char_tokenize(strings: list[str], block_size: int):
    """
    Tokenize the list of strings by building blocks and also return a translator from character to index
    """
    vocab = sorted(list(set(".".join(strings))))
    str_to_int = stoi(vocab)
    x_tokens = []
    y_tokens = []
    for string in strings:
        context = [0] * block_size
        for s in string + ".":
            ix = str_to_int[s]
            x_tokens.append(context)
            y_tokens.append(ix)
            context = context[1:] + [ix]

    return np.array(x_tokens), np.array(y_tokens) , vocab

def stoi(vocab: list[str]) -> dict[str, int]:
    """
    Build string to int converter
    """
    return {v: i for i, v in enumerate(vocab)}

def itos(vocab: list[str]) -> dict[str, int]:
    """
    Build int to string converter
    """
    return dict(enumerate(vocab))
