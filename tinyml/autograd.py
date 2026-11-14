"""A small reverse-mode automatic differentiation engine (Chapter 11).

A Tensor wraps a NumPy array. Every operation on tensors records its inputs
and a function that maps the gradient of the output to gradients of the
inputs. Calling backward() on a scalar result then computes the gradient of
that result with respect to every tensor that requires one.
"""
import numpy as np


def unbroadcast(grad, shape):
    """Sums grad down to shape, undoing NumPy broadcasting.

    If b of shape (3,) was broadcast against x of shape (4, 3), the gradient
    that arrives for b has shape (4, 3). Each element of b was used four
    times, so its gradient is the sum over the broadcast axis.
    """
    while grad.ndim > len(shape):            # axes that broadcasting prepended
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):      # axes stretched from size 1
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def as_tensor(x):
    """Wraps numbers and arrays as constant Tensors; passes Tensors through."""
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """An array that remembers how it was computed."""

    __array_ufunc__ = None   # makes ndarray + Tensor call Tensor.__radd__

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None          # filled in by backward()
        self._parents = ()        # the tensors this one was computed from
        self._backward = None     # maps out.grad to a tuple of parent grads
        self._op = ""             # the operation's name, for debugging

    @classmethod
    def from_op(cls, data, parents, backward, op=""):
        """Creates the output of an operation. Other modules use this to add
        new operations, such as the convolution in Chapter 12."""
        out = cls(data)
        if any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = tuple(parents)
            out._backward = backward
            out._op = op
        return out

    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    size = property(lambda self: self.data.size)

    def item(self):
        return self.data.item()

    def detach(self):
        """Returns a constant copy that gradients will not flow through."""
        return Tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4)}{flag})"

    def backward(self, grad=None):
        """Computes d(self)/d(t) for every tensor t that self depends on.

        Gradients of leaf tensors (parameters and inputs) are added to their
        .grad, so call zero_grad() between training steps. Intermediate
        tensors get their .grad overwritten, which is handy for inspection.
        """
        if grad is None:
            if self.size != 1:
                raise ValueError("backward() without a gradient needs a scalar")
            grad = np.ones_like(self.data)

        # Topological order: every tensor appears after all of its parents.
        # An explicit stack avoids Python's recursion limit on deep graphs.
        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
            elif id(node) not in visited:
                visited.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents
                             if id(p) not in visited)

        # Walk the graph from the output back to the leaves (the chain rule).
        grads = {id(self): np.asarray(grad, dtype=float)}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None or not node.requires_grad:
                continue
            if node._backward is None:                    # a leaf
                node.grad = np.array(g) if node.grad is None else node.grad + g
                continue
            node.grad = g
            for parent, pg in zip(node._parents, node._backward(g)):
                if pg is None or not parent.requires_grad:
                    continue
                pg = unbroadcast(np.asarray(pg, dtype=float), parent.shape)
                key = id(parent)
                grads[key] = pg if key not in grads else grads[key] + pg

    def __add__(self, other):
        other = as_tensor(other)
        return Tensor.from_op(self.data + other.data, (self, other),
                              lambda g: (g, g), "+")

    def __mul__(self, other):
        other = as_tensor(other)
        a, b = self.data, other.data
        return Tensor.from_op(a * b, (self, other),
                              lambda g: (g * b, g * a), "*")

    def __neg__(self):
        return Tensor.from_op(-self.data, (self,), lambda g: (-g,), "neg")

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __truediv__(self, other):
        other = as_tensor(other)
        a, b = self.data, other.data
        return Tensor.from_op(a / b, (self, other),
                              lambda g: (g / b, -g * a / b**2), "/")

    def __pow__(self, p):
        """Raises to a constant power p (a number, not a Tensor)."""
        a = self.data
        return Tensor.from_op(a ** p, (self,),
                              lambda g: (g * p * a ** (p - 1),), "**")

    def __matmul__(self, other):
        """Matrix product of operands with at least two dimensions. Leading
        dimensions are batch dimensions and may broadcast."""
        other = as_tensor(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("@ needs operands with at least 2 dimensions")
        def backward(g):
            return g @ np.swapaxes(b, -1, -2), np.swapaxes(a, -1, -2) @ g
        return Tensor.from_op(a @ b, (self, other), backward, "@")

    # Reflected versions, so that 2 * x and 1 - x work.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return as_tensor(other) - self

    def __rtruediv__(self, other):
        return as_tensor(other) / self

    def __rmatmul__(self, other):
        return as_tensor(other) @ self

    def sum(self, axis=None, keepdims=False):
        shape = self.shape
        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape),)
        return Tensor.from_op(self.data.sum(axis=axis, keepdims=keepdims),
                              (self,), backward, "sum")

    def mean(self, axis=None, keepdims=False):
        total = self.sum(axis=axis, keepdims=keepdims)
        return total * (total.size / self.size)

    def max(self, axis=None, keepdims=False):
        """Maximum along axis. Tied maxima share the gradient equally."""
        a = self.data
        m = a.max(axis=axis, keepdims=True)
        mask = (a == m).astype(float)
        mask /= mask.sum(axis=axis, keepdims=True)
        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (mask * g,)
        out = m if keepdims else a.max(axis=axis)
        return Tensor.from_op(out, (self,), backward, "max")

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.shape
        return Tensor.from_op(self.data.reshape(shape), (self,),
                              lambda g: (g.reshape(old),), "reshape")

    def transpose(self, *axes):
        """Permutes axes; with no arguments, reverses them like ndarray.T."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return Tensor.from_op(self.data.transpose(axes), (self,),
                              lambda g: (g.transpose(inverse),), "transpose")

    T = property(lambda self: self.transpose())

    def __getitem__(self, index):
        """Indexing and slicing. Repeated indices accumulate gradient."""
        if isinstance(index, Tensor):
            index = index.data.astype(int)
        shape = self.shape
        def backward(g):
            full = np.zeros(shape)
            np.add.at(full, index, g)
            return (full,)
        return Tensor.from_op(self.data[index], (self,), backward, "index")

    def exp(self):
        out = np.exp(self.data)
        return Tensor.from_op(out, (self,), lambda g: (g * out,), "exp")

    def log(self):
        a = self.data
        return Tensor.from_op(np.log(a), (self,), lambda g: (g / a,), "log")

    def relu(self):
        a = self.data
        return Tensor.from_op(np.maximum(a, 0.0), (self,),
                              lambda g: (g * (a > 0),), "relu")

    def tanh(self):
        out = np.tanh(self.data)
        return Tensor.from_op(out, (self,),
                              lambda g: (g * (1.0 - out**2),), "tanh")

    def sigmoid(self):
        a = self.data
        # Two formulas, each used where it cannot overflow.
        e = np.exp(-np.abs(a))
        out = np.where(a >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        return Tensor.from_op(out, (self,),
                              lambda g: (g * out * (1.0 - out),), "sigmoid")


def check_grad(f, inputs, eps=1e-6):
    """Compares backward() with central finite differences.

    f maps a list of Tensors to a scalar Tensor. Returns the largest relative
    error over all input elements; values below about 1e-6 mean agreement.
    """
    for t in inputs:
        t.grad = None
    f(inputs).backward()
    worst = 0.0
    for t in inputs:
        numeric = np.zeros_like(t.data)
        for i in np.ndindex(t.shape):
            saved = t.data[i]
            t.data[i] = saved + eps
            up = f(inputs).item()
            t.data[i] = saved - eps
            down = f(inputs).item()
            t.data[i] = saved
            numeric[i] = (up - down) / (2 * eps)
        analytic = t.grad if t.grad is not None else np.zeros_like(t.data)
        scale = np.maximum(1.0, np.abs(numeric) + np.abs(analytic))
        worst = max(worst, float(np.max(np.abs(numeric - analytic) / scale)))
    return worst
