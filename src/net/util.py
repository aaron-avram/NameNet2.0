"""
File containing utility functions
"""
import numpy as np

def unbroadcast(arr, target_shape: tuple) -> np.ndarray:
    """
    Unbroadcast array to target shape by summing along target axes,
    assuming the target shape and array shape are compatible
    """
    while len(arr.shape) > len(target_shape):
        arr = np.sum(arr, axis=0)
    for i, (acc, target) in enumerate(zip(arr.shape, target_shape)):
        if target == 1 and acc != 1:
            arr = np.sum(arr, axis=i, keepdims=True)
    return arr
