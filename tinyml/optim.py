"""Optimizers that update Tensor parameters in place (Chapter 11)."""
import numpy as np


class Optimizer:
    """Holds a list of parameters. Subclasses implement step()."""

    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def _grads(self, weight_decay):
        """Yields (index, parameter, gradient), adding L2 weight decay."""
        for i, p in enumerate(self.params):
            if p.grad is not None:
                yield i, p, p.grad + weight_decay * p.data


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

    v <- momentum * v + g, then p <- p - lr * v. With momentum=0 this is
    plain gradient descent on the mini-batch gradient g.
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p, g in self._grads(self.weight_decay):
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]


class Adam(Optimizer):
    """Adam (Kingma and Ba, 2015): per-parameter step sizes from running
    averages of the gradient (m) and of its square (v), bias-corrected.

    weight_decay adds an L2 penalty to the gradient. With decoupled=True it
    instead shrinks the weights directly, which is the AdamW variant.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=False):
        super().__init__(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay, self.decoupled = weight_decay, decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        coupled = 0.0 if self.decoupled else self.weight_decay
        for i, p, g in self._grads(coupled):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            if self.decoupled:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(params, max_norm):
    """Rescales gradients so their combined L2 norm is at most max_norm.

    Returns the norm before clipping, which is worth logging: a sudden jump
    is often the first sign of exploding gradients.
    """
    grads = [p.grad for p in params if p.grad is not None]
    norm = float(np.sqrt(sum((g ** 2).sum() for g in grads)))
    if norm > max_norm:
        for g in grads:
            g *= max_norm / (norm + 1e-12)
    return norm
