"""Principal component analysis (Chapter 10)."""
import numpy as np

from tinyml.base import Transformer


class PCA(Transformer):
    """Projects data onto its directions of greatest variance.

    The components are the right singular vectors of the centered data.
    n_components=None keeps all of them. Each component's sign is fixed so
    that its largest-magnitude loading is positive, which makes results
    reproducible (an SVD may return either sign).
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        biggest = np.abs(Vt).argmax(axis=1)
        Vt *= np.sign(Vt[np.arange(len(Vt)), biggest])[:, None]
        k = self.n_components or len(S)
        variance = S ** 2 / (n - 1)                  # variance along each direction
        self.components_ = Vt[:k]                    # one row per component
        self.singular_values_ = S[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / variance.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        """Maps component scores back to the original space. With fewer than
        all components, this is the best rank-k approximation of X."""
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_
