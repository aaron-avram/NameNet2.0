"""
File containing the tensor class
"""

import numpy as np

class Tensor:
    """
    A class that supports tensor like operations combined with gradient tracking
    """

    value: np.ndarray
    grad: np.ndarray
    parents: tuple
    op: str
    shape: tuple
    grad_required: bool

    def __init__(self, value: np.ndarray | int | float, parents: set = None, op: str = None, grad_required: bool = True):
        if isinstance(value, np.ndarray):
            self.value = value
            self.shape = value.shape
        if isinstance(value, (int, float)):
            self.value = np.ndarray(float(value)) # Wrap scalars
            self.shape = self.value.shape
        self.parents = parents
        self.op = op
        self.grad_required = grad_required
        if grad_required:
            self.grad = np.zeros_like(value)
        else:
            self.grad = None

    def __matmul__(self, other):
        """
        Matrix multiply on the left by tensor to create a new Tensor
        """
        if isinstance(other, Tensor):
            return Tensor(self.value @ other.value, parents=(self, other), op='@', grad_required=(self.grad_required or other.grad_required))
        if isinstance(other, np.ndarray):
            other = Tensor(other, grad_required=self.grad_required)
            return self @ other
        return NotImplemented

    def __rmatmul__(self, other):
        """
        Matrix multiply on the right by tensor to create a new Tensor
        """
        if isinstance(other, Tensor):
            return other @ self
        if isinstance(other, np.ndarray):
            other = Tensor(other, grad_required=self.grad_required)
            return other @ self
        return NotImplemented

    def __add__(self, other):
        """
        Add on the left by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return Tensor(self.value + other.value, parents=(self, other), op='+', grad_required=(self.grad_required or other.grad_required))
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, grad_required=self.grad_required)
            return self + other
        return NotImplemented

    def __radd__(self, other):
        """
        Add on the right by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return self + other
        if isinstance(other, (np.ndarray, int, float)):
            return self + other
        return NotImplemented
    
    def __sub__(self, other):
        """
        Subtract on the left from tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return Tensor(self.value - other.value, parents=(self, other), op='-', grad_required=(self.grad_required or other.grad_required))
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, self.grad_required)
            return self - other
        return NotImplemented

    def __rsub__(self, other):
        """
        Subtract on the right by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return other - self
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, grad_required=self.grad_required)
            return other - self
        return NotImplemented

    def __mul__(self, other):
        """
        Multiply on the left by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return Tensor(self.value * other.value, parents=(self, other), op='*', grad_required=(self.grad_required or other.grad_required))
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, grad_required=self.grad_required)
            return self * other
        return NotImplemented

    def __rmul__(self, other):
        """
        Multiply on the right by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return other * self
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, grad_required=self.grad_required)
            return other * self
        return NotImplemented

    def __div__(self, other):
        """
        Divide on the left by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return Tensor(self.value / other.value, parents=(self, other), op='/', grad_required=(self.grad_required or other.grad_required))
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, grad_required=self.grad_required)
            return self / other
        return NotImplemented

    def __rdiv__(self, other):
        """
        Divide on the right by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return other / self
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(self, grad_required=self.grad_required)
            return other / self
        return NotImplemented

    def __getitem__(self, idx):
        """
        Index into tensor
        """
        if isinstance(idx, Tensor):
            return self.value[idx.value]
        if isinstance(idx, (np.ndarray, int)):
            return self.value[idx]
        return NotImplemented

    def __setitem__(self, idx, item):
        """
        Mutate tensor
        """
        self.value[idx] = item.value if isinstance(item, Tensor) else item

    def __array__(self):
        """
        Return array representation of tensor
        """
        return self.value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercept universal function and make compatible with Tensor class
        """
        raw_inputs = [x.value if isinstance(x, Tensor) else x for x in inputs]
        result = getattr(ufunc, method)(*raw_inputs, **kwargs)
        return Tensor(result, self.grad_required)

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercept higher level operations and make compatible with Tensor class
        """
        if not all(issubclass(t, Tensor) for t in types):
            return NotImplemented
        raw_args = [x.value if isinstance(x, Tensor) else x for x in args]
        result = func(*raw_args, **kwargs)
        return Tensor(result, self.grad_required)

    def tanh(self):
        """
        Tanh activation function
        """
        return Tensor(np.tanh(self.value), parents=(self,), op='tanh', grad_required=self.grad_required)

    def relu(self):
        """
        ReLU activation function
        """
        return Tensor(np.maximum(self, 0), parents=(self), op='relu', grad_required=self.grad_required)
    
    def cross_entropy(self, targets):
        """
        Cross entropy loss for tensor assuming it is a tensor of logits
        """
        logits = self.value - np.max(self.value, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Clip probs
        eps = 1e-12
        np.clip(probs, eps, 1 - eps)

        n = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(n), targets]))
        return Tensor(loss, parents=(self, targets), op='cross_entropy', grad_required=self.grad_required)