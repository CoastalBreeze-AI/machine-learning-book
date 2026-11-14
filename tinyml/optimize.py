"""Gradient-based optimizers and a gradient checker (Chapter 4).

An objective is a function that takes a parameter vector w and returns the
pair (loss, gradient). Computing both in one function lets them share work,
and lets every optimizer here accept any objective with that shape.
"""
import numpy as np


def gradient_descent(objective, w0, lr=0.1, n_iter=100, momentum=0.0, tol=0.0):
    """Minimizes objective(w) -> (loss, grad), starting from w0.

    Each step updates a velocity v = momentum * v - lr * grad and then sets
    w = w + v. With momentum=0 this is plain gradient descent; with
    0 < momentum < 1 it is Polyak's heavy-ball method. The loop stops early
    if the gradient's norm drops to tol or the loss stops being finite.

    Returns (w, history), where history[k] is the loss after k steps.
    """
    w = np.array(w0, dtype=float)
    velocity = np.zeros_like(w)
    loss, grad = objective(w)
    history = [loss]
    for _ in range(n_iter):
        if not np.isfinite(loss) or np.linalg.norm(grad) <= tol:
            break
        velocity = momentum * velocity - lr * grad
        w = w + velocity
        loss, grad = objective(w)
        history.append(loss)
    return w, history


def minibatch_sgd(objective, w0, X, y, lr=0.01, n_epochs=10, batch_size=32,
                  momentum=0.0, seed=0):
    """Minimizes a loss over (X, y) using gradients from random mini-batches.

    objective(w, X_batch, y_batch) -> (loss, grad) must return the loss and
    gradient averaged over the rows it is given. Each epoch shuffles the
    rows, then takes one step per batch. batch_size=1 gives classic
    stochastic gradient descent; batch_size=len(y) gives gradient descent.

    Returns (w, history), where history[e] is the full-data loss after e
    epochs. Computing it costs one extra pass per epoch, for monitoring only.
    """
    rng = np.random.default_rng(seed)
    w = np.array(w0, dtype=float)
    velocity = np.zeros_like(w)
    history = [objective(w, X, y)[0]]
    for _ in range(n_epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            _, grad = objective(w, X[batch], y[batch])
            velocity = momentum * velocity - lr * grad
            w = w + velocity
        history.append(objective(w, X, y)[0])
        if not np.isfinite(history[-1]):
            break
    return w, history


def numerical_gradient(f, w, eps=1e-6):
    """Estimates the gradient of a scalar function f at w by central differences."""
    w = np.array(w, dtype=float)
    grad = np.zeros_like(w)
    for i in range(w.size):
        step = np.zeros_like(w)
        step.flat[i] = eps
        grad.flat[i] = (f(w + step) - f(w - step)) / (2 * eps)
    return grad


def check_gradient(objective, w, eps=1e-6):
    """Compares the analytic gradient of objective(w) -> (loss, grad) with a
    numerical estimate, and returns their relative error.

    The relative error is |g - g_num| / (|g| + |g_num|), using Euclidean
    norms. For a correct gradient in double precision it is usually below
    1e-7; values above about 1e-4 almost always mean a bug.
    """
    _, analytic = objective(w)
    numeric = numerical_gradient(lambda v: objective(v)[0], w, eps)
    scale = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if scale == 0.0:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / scale)
