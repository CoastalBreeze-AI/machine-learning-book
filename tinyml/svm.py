"""Kernels and a kernel support vector machine (Chapter 7)."""
import numpy as np

from tinyml.base import Estimator


def linear_kernel(A, B):
    """k(a, b) = a . b"""
    return np.asarray(A, dtype=float) @ np.asarray(B, dtype=float).T


def polynomial_kernel(A, B, degree=3, coef0=1.0, gamma=1.0):
    """k(a, b) = (gamma * a . b + coef0) ** degree"""
    return (gamma * linear_kernel(A, B) + coef0) ** degree


def rbf_kernel(A, B, gamma=1.0):
    """k(a, b) = exp(-gamma * ||a - b||^2), the Gaussian or RBF kernel."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    sq = (A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]
    return np.exp(-gamma * np.maximum(sq, 0.0))


def make_kernel(kernel, gamma, degree, coef0):
    """Turns a kernel name and its settings into a function of (A, B)."""
    if callable(kernel):
        return kernel
    if kernel == "linear":
        return linear_kernel
    if kernel == "rbf":
        return lambda A, B: rbf_kernel(A, B, gamma)
    if kernel == "poly":
        return lambda A, B: polynomial_kernel(A, B, degree, coef0, gamma)
    raise ValueError(f"unknown kernel '{kernel}'")


def smo(K, y, C, tol=1e-3, max_iter=100_000):
    """Solves the soft-margin SVM dual by sequential minimal optimization.

    Minimizes 0.5 a'Qa - sum(a) with Q_ij = y_i y_j K_ij, subject to
    0 <= a_i <= C and y'a = 0, where y holds -1 and +1. Each step changes
    one pair (i, j) along a direction that keeps y'a = 0 (Section 7.7).
    Returns (alpha, b, n_iter).
    """
    n = len(y)
    alpha = np.zeros(n)
    G = -np.ones(n)  # gradient of the dual objective, Q @ alpha - 1
    diag = np.diag(K)
    for it in range(max_iter):
        # Indices that may move "up" (in the direction of y) or "down".
        up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
        low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
        score = -y * G
        i = np.flatnonzero(up)[np.argmax(score[up])]
        m, M = score[i], score[low].min()
        if m - M < tol:  # the KKT conditions hold to within tol
            break
        # Second-order choice of j: the largest guaranteed decrease.
        candidates = np.flatnonzero(low & (score < m))
        curv = np.maximum(diag[i] + diag[candidates] - 2 * K[i, candidates], 1e-12)
        gap = m - score[candidates]
        j = candidates[np.argmax(gap ** 2 / curv)]
        eta = max(diag[i] + diag[j] - 2 * K[i, j], 1e-12)
        t = (m - score[j]) / eta  # unconstrained step along the pair
        # Clip t so that both alphas stay inside the box [0, C].
        t = min(t, C - alpha[i] if y[i] > 0 else alpha[i])
        t = min(t, alpha[j] if y[j] > 0 else C - alpha[j])
        alpha[i] += y[i] * t
        alpha[j] -= y[j] * t
        G += t * y * (K[i] - K[j])  # K is symmetric; rows are fast to read
    score = -y * G
    free = (alpha > 1e-8) & (alpha < C - 1e-8)
    b = score[free].mean() if free.any() else (m + M) / 2
    return alpha, b, it + 1


class KernelSVM(Estimator):
    """A binary soft-margin support vector machine with any kernel.

    decision_function returns sum_i alpha_i y_i k(x_i, x) + b, whose sign is
    the prediction. This class does not estimate probabilities; Exercise 9
    adds Platt scaling.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, degree=3, coef0=1.0,
                 tol=1e-3, max_iter=100_000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("KernelSVM handles two classes only")
        signs = np.where(y == self.classes_[1], 1.0, -1.0)
        self.kernel_ = make_kernel(self.kernel, self.gamma, self.degree, self.coef0)
        K = self.kernel_(X, X)  # n x n: fine for a few thousand rows
        alpha, self.intercept_, self.n_iter_ = smo(K, signs, self.C,
                                                   self.tol, self.max_iter)
        support = alpha > 1e-8  # only these points affect predictions
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.dual_coef_ = alpha[support] * signs[support]  # alpha_i * y_i
        return self

    def decision_function(self, X):
        K = self.kernel_(np.asarray(X, dtype=float), self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
