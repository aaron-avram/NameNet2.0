"""
File containing the tensor class
"""

# Imports
import numpy as np
from net.util import unbroadcast

# PyLint Configs
# pylint: disable=protected-access,attribute-defined-outside-init

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

    def __init__(self, value: np.ndarray | int | float, parents: tuple=None, op: str = None, grad_required: bool = True):
        if isinstance(value, np.ndarray):
            if grad_required:
                self.value = value.astype(np.float64, copy=False)
            else:
                self.value = value
            self.shape = value.shape

        elif isinstance(value, (int, float)):
            self.value = np.array(value, dtype=np.float64) # Wrap scalars
            self.shape = self.value.shape
        else:
            raise NotImplementedError
        self.parents = parents if parents is not None else ()
        self.op = op
        self.grad_required = grad_required
        if grad_required:
            self.grad = np.zeros_like(self.value)
        else:
            self.grad = None
    
    def __len__(self):
        """ Len function """
        return len(self.value)

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

    def _matmul_backward(self):
        assert self.op == '@'
        a, b = self.parents
        dc = self.grad
        # Reshape 1D tensors to 2D row vectors
        if a.value.ndim == 1:
            a_val = a.value.reshape(1, -1)
        else:
            a_val = a.value

        if b.value.ndim == 1:
            b_val = b.value.reshape(1, -1)
        else:
            b_val = b.value

        if len(dc.shape) == 1:
            dc_val = dc.reshape(1, -1)
        else:
            dc_val = dc
        da = dc_val @ b_val.T
        db = a_val.T @ dc_val

        return (unbroadcast(da, a.shape), unbroadcast(db, b.shape))

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

    def __iadd__(self, other):
        """In place addition"""
        if isinstance(other, Tensor):
            self.value = self.value + other.value
            return self
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, self.grad_required)
            self.value = self.value + other
            return self
        return NotImplemented

    def _add_backward(self):
        assert self.op == '+'
        return (unbroadcast(self.grad, self.parents[0].shape),
                unbroadcast(self.grad, self.parents[1].shape))

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

    def __isub__(self, other):
        """In place subtraction"""
        if isinstance(other, Tensor):
            self.value = self.value - other.value
            return self
        if isinstance(other, (np.ndarray, int, float)):
            self.value = self.value - other
            return self
        return NotImplemented

    def _sub_backward(self):
        assert self.op == '-'
        return (unbroadcast(self.grad, self.parents[0].shape),
                unbroadcast(self.grad, -self.parents[1].shape))

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

    def _mul_backward(self):
        assert self.op == '*'
        return (unbroadcast(self.grad * self.parents[1].value, self.parents[0].shape),
                unbroadcast(self.grad * self.parents[0].value, self.parents[1].shape))

    def __truediv__(self, other):
        """
        Divide on the left by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return Tensor(self.value / other.value, parents=(self, other), op='/', grad_required=(self.grad_required or other.grad_required))
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(other, grad_required=self.grad_required)
            return self / other
        return NotImplemented

    def __rtruediv__(self, other):
        """
        Divide on the right by tensor and return a new Tensor
        """
        if isinstance(other, Tensor):
            return other / self
        if isinstance(other, (np.ndarray, int, float)):
            other = Tensor(self, grad_required=self.grad_required)
            return other / self
        return NotImplemented

    def __lt__(self, other):
        """
        Less than
        """
        if isinstance(other, Tensor):
            return self.value < other.value
        if isinstance(other, np.ndarray | int | float):
            return self.value < other
        return NotImplemented

    def __le__(self, other):
        """
        Less than or equal to
        """
        if isinstance(other, Tensor):
            return self.value <= other.value
        if isinstance(other, np.ndarray | int | float):
            return self.value <= other
        return NotImplemented

    def __eq__(self, other):
        """
        Equal to
        """
        if isinstance(other, Tensor):
            return self.value == other.value
        if isinstance(other, np.ndarray | int | float):
            return self.value == other
        return NotImplemented
    def __gt__(self, other):
        """
        Less than
        """
        if isinstance(other, Tensor):
            return self.value > other.value
        if isinstance(other, np.ndarray | int | float):
            return self.value > other
        return NotImplemented

    def __ge__(self, other):
        """
        Less than or equal to
        """
        if isinstance(other, Tensor):
            return self.value >= other.value
        if isinstance(other, np.ndarray | int | float):
            return self.value >= other
        return NotImplemented

    def __ne__(self, other):
        """
        Equal to
        """
        if isinstance(other, Tensor):
            return self.value != other.value
        if isinstance(other, np.ndarray | int | float):
            return self.value != other
        return NotImplemented

    def __getitem__(self, idx):
        """
        Return a non-differentiable view of the underlying data
        """
        if isinstance(idx, Tensor):
            idx = idx.value
        if isinstance(idx, (np.ndarray, int, slice, tuple)):
            return Tensor(self.value[idx], grad_required=False)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __setitem__(self, idx, item):
        """
        In-place mutation of tensor data
        """
        item_val = item.value if isinstance(item, Tensor) else item
        self.value[idx] = item_val

    def __array__(self, dtype=None):
        """
        Return array representation of tensor
        """
        return np.asarray(self.value, dtype=dtype)

    @staticmethod
    def _unwrap(x):
        if isinstance(x, Tensor):
            return x.value
        elif isinstance(x, (list, tuple)):
            return type(x)(Tensor._unwrap(e) for e in x)
        return x
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
        if not any(issubclass(t, Tensor) for t in types):
            return NotImplemented
        raw_args = Tensor._unwrap(args)
        result = func(*raw_args, **kwargs)
        return Tensor(result, self.grad_required)

    def tanh(self):
        """
        Tanh activation function
        """
        return Tensor(np.tanh(self.value), parents=(self,), op='tanh', grad_required=self.grad_required)

    def _tanh_backward(self):
        assert self.op == 'tanh'
        return ((1 - self.value ** 2) * self.grad,)

    def relu(self):
        """
        ReLU activation function
        """
        return Tensor(np.maximum(0, self.value), parents=(self, ), op='relu', grad_required=self.grad_required)

    def _relu_backward(self):
        assert self.op == 'relu'
        return ((self.value > 0).astype(float) * self.grad,)

    def cross_entropy(self, targets):
        """
        Cross entropy loss for tensor assuming it is a tensor of logits
        """
        logits = self.value - np.max(self.value, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Clip probs
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)

        n = logits.shape[0]
        targ_vals = np.asarray(targets, dtype=int)
        loss = -np.mean(np.log(probs[np.arange(n), targ_vals]))
        return Tensor(loss, parents=(self, targets), op='cross_entropy', grad_required=self.grad_required)

    def _cross_entropy_backward(self):
        assert self.op == 'cross_entropy'
        vals, targets = self.parents
        logits = vals - np.max(vals, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        grad = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Clip probs
        eps = 1e-12
        grad = np.clip(grad, eps, 1 - eps)

        # Assemble grad
        n = logits.shape[0]
        targ_vals = np.asarray(targets, dtype=int)
        grad[np.arange(n), targ_vals] -= 1
        grad /= n
        return (grad, None)

    def transpose(self, axes= None):
        """
        Transpose tensor with given axes
        """
        out = Tensor(self.value.transpose() if axes is None else self.value.transpose(axes),
                     parents=(self,), op='T', grad_required=self.grad_required)
        out._transpose_axes = axes
        return out

    @property
    def T(self):
        """
        Shorthand for transpose
        """
        return self.transpose()

    def _transpose_backward(self):
        assert self.op == 'T'
        axes =  getattr(self, '_transpose_axes', None)
        if axes is None:
            return (self.grad.T,)
        inv_axes = np.argsort(axes)
        return (self.grad.transpose(inv_axes),)

    def reshape(self, shape: tuple):
        """
        Reshape tensor and return a new one with the given shape
        """
        out = Tensor(np.reshape(self.value, shape=shape),
                    parents = (self,), op='reshape', grad_required=self.grad_required)
        return out

    def _reshape_backward(self):
        assert self.op == 'reshape'
        return (self.grad.reshape(shape=self.shape),)

    def sum(self, axis=None, keepdims=False):
        """
        Sum along given axis
        """
        out = Tensor(np.sum(self.value, axis=axis, keepdims=keepdims), parents=(self,),
                    op='sum', grad_required=self.grad_required)
        out._sum_axis = axis
        out._sum_keepdims = keepdims
        return out

    def _sum_backward(self):
        assert self.op == 'sum'
        parent = self.parents[0]
        grad = self.grad

        axis = getattr(self, '_sum_axis', None)
        keepdims = getattr(self, '_sum_keepdims', False)

        if not keepdims and axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                np.expand_dims(grad, ax)
        return np.broadcast_to(grad, parent.shape)

    def embed(self, emb_matrix):
        """
        Embed Tensor using emb_matrix
        """
        if isinstance(emb_matrix, Tensor):
            out = emb_matrix.value[self.value]
            out = np.reshape(out, shape=(out.shape[0], -1))
            return Tensor(out, parents=(self, emb_matrix), op='emb', grad_required=(self.grad_required or emb_matrix.grad_required))
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = Tensor(emb_matrix)
            return self.embed(emb_matrix)
        return NotImplemented

    def _embed_backward(self):
        assert self.op == 'emb'
        idxs, emb_matrix = self.parents
        out_grad = np.zeros_like(emb_matrix)
        flat_idxs = np.ravel(idxs.value)

        upd_grad = np.reshape(self.grad, shape=(-1, emb_matrix.shape[1]))
        np.add.at(out_grad, flat_idxs, upd_grad)
        return out_grad



    def zero_grad_shallow(self):
        """
        Reset gradient
        """
        self.grad = np.zeros_like(self.value)
    def zero_grad_deep(self):
        """
        Reset gradient of all parents
        """
        self.zero_grad_shallow()
        for p in self.parents:
            p.zero_grad_deep()

    def item(self):
        """
        Get item of tensor if tensor is a scalar
        """
        if self.shape == ():
            return self.value.item()
        raise ValueError
    
    def clip_grad(self, max_norm: float):
        """ Clip grad """
        grad_norm = np.linalg.norm(self.grad)
        if grad_norm > max_norm:
            self.grad *= (max_norm / grad_norm)

    def _local_grads(self):
        """
        Get local gradients
        """
        if self.op is None:
            return ()
        if self.op == 'cross_entropy':
            return self._cross_entropy_backward()
        if self.op == 'tanh':
            return self._tanh_backward()
        if self.op == 'relu':
            return self._relu_backward()
        if self.op == '@':
            return self._matmul_backward()
        if self.op == '+':
            return self._add_backward()
        if self.op == '-':
            return self._sub_backward()
        if self.op == '*':
            return self._mul_backward()
        if self.op == 'T':
            return self._transpose_backward()
        if self.op == 'reshape':
            return self._reshape_backward()
        if self.op == 'sum':
            return self._sum_backward()
        return NotImplemented

    def backward(self, grad=None):
        """
        Perform backward pass starting at the current node
        """
        if grad is None:
            grad = np.ones_like(self.value)
        if self.grad is None:
            self.grad = np.zeros_like(self.value)
        self.grad += grad

        for parent, parent_grad in zip(self.parents, self._local_grads()):
            if parent_grad is not None:
                parent.backward(parent_grad)
