"""Bayesian linear regression (Chapter 9).

The prior on the weights is w ~ N(0, I / alpha), and the noise on each target
is N(0, 1 / beta). Both are Gaussian, so the posterior over w is Gaussian too,
and so is the predictive distribution for a new input.
"""
import numpy as np

from tinyml.base import Estimator


class BayesianLinearRegression(Estimator):
    """Linear regression that returns a distribution over weights.

    alpha is the prior precision of each weight and beta the noise precision
    (1 / noise variance). The intercept is treated as one more weight with
    the same prior, so center y or scale alpha accordingly.
    """

    def __init__(self, alpha=1.0, beta=1.0, fit_intercept=True):
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X

    def fit(self, X, y):
        Phi, y = self._design(X), np.asarray(y, dtype=float)
        precision = self.alpha * np.eye(Phi.shape[1]) + self.beta * Phi.T @ Phi
        self.cov_ = np.linalg.inv(precision)                 # S_N
        self.mean_ = self.beta * self.cov_ @ Phi.T @ y       # m_N
        if self.fit_intercept:
            self.intercept_, self.coef_ = self.mean_[0], self.mean_[1:]
        else:
            self.intercept_, self.coef_ = 0.0, self.mean_
        return self

    def predict(self, X, return_std=False):
        """Predictive mean, and optionally the predictive standard deviation:
        var = 1 / beta (noise) + phi^T S_N phi (uncertainty about w)."""
        Phi = self._design(X)
        mean = Phi @ self.mean_
        if not return_std:
            return mean
        weight_var = np.sum((Phi @ self.cov_) * Phi, axis=1)
        return mean, np.sqrt(1.0 / self.beta + weight_var)

    def sample_weights(self, n_samples, seed=0):
        """Draws weight vectors from the posterior, one per row."""
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mean_, self.cov_, size=n_samples)

    def log_evidence(self, X, y):
        """log p(y | X, alpha, beta), with the weights integrated out.
        Call it after fit() on the same X and y (Bishop, eq. 3.86)."""
        Phi, y = self._design(X), np.asarray(y, dtype=float)
        n, m = Phi.shape
        fit = self.beta / 2 * np.sum((y - Phi @ self.mean_) ** 2)
        size = self.alpha / 2 * self.mean_ @ self.mean_
        _, logdet_cov = np.linalg.slogdet(self.cov_)     # = -log det(precision)
        return (m / 2 * np.log(self.alpha) + n / 2 * np.log(self.beta)
                - fit - size + logdet_cov / 2 - n / 2 * np.log(2 * np.pi))
