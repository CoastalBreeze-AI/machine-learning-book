"""Nearest-neighbor models (Chapter 7).

A nearest-neighbor model stores the training set and does its work at
prediction time, by finding the training points closest to each query.
"""
import numpy as np

from tinyml.base import Estimator


def pairwise_distances(A, B, metric="euclidean"):
    """Returns the (len(A), len(B)) matrix of distances between rows.

    Euclidean distances use ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, so the
    work is one matrix product instead of a Python loop over pairs.
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if metric == "euclidean":
        sq = (A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]
        return np.sqrt(np.maximum(sq, 0.0))  # rounding can make tiny values negative
    if metric == "manhattan":
        D = np.zeros((len(A), len(B)))
        for j in range(A.shape[1]):  # one feature at a time keeps memory small
            D += np.abs(A[:, j, None] - B[None, :, j])
        return D
    if metric == "cosine":
        norms = np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :]
        return 1.0 - (A @ B.T) / np.maximum(norms, 1e-12)
    raise ValueError(f"unknown metric '{metric}'")


def nearest_neighbors(X_train, X_query, k, metric="euclidean", batch_size=500):
    """Finds the k nearest training rows for each query row.

    Returns (distances, indices), each of shape (len(X_query), k), sorted
    from nearest to farthest. Queries are processed in batches so that the
    distance matrix never needs more than batch_size * len(X_train) floats.
    """
    k = min(k, len(X_train))
    all_dist, all_index = [], []
    for start in range(0, len(X_query), batch_size):
        D = pairwise_distances(X_query[start:start + batch_size], X_train, metric)
        # argpartition finds the k smallest in linear time; then sort just those.
        index = np.argpartition(D, k - 1, axis=1)[:, :k]
        dist = np.take_along_axis(D, index, axis=1)
        order = np.argsort(dist, axis=1, kind="stable")
        all_index.append(np.take_along_axis(index, order, axis=1))
        all_dist.append(np.take_along_axis(dist, order, axis=1))
    return np.vstack(all_dist), np.vstack(all_index)


def _neighbor_weights(dist, weights):
    """Uniform weights, or inverse-distance weights with exact matches dominating."""
    if weights == "uniform":
        return np.ones_like(dist)
    if weights == "distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        exact = np.isinf(w)
        rows = exact.any(axis=1)  # a query that coincides with training points
        w[rows] = exact[rows].astype(float)  # uses only those points
        return w
    raise ValueError(f"unknown weights '{weights}'")


class KNNClassifier(Estimator):
    """Predicts by a (weighted) vote among the k nearest training points."""

    def __init__(self, k=5, weights="uniform", metric="euclidean"):
        self.k = k
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_index_ = np.unique(y, return_inverse=True)
        return self  # "training" is just remembering the data

    def predict_proba(self, X):
        dist, index = nearest_neighbors(self.X_, np.asarray(X, dtype=float),
                                        self.k, self.metric)
        w = _neighbor_weights(dist, self.weights)
        votes = np.zeros((len(index), len(self.classes_)))
        for c in range(len(self.classes_)):
            votes[:, c] = (w * (self.y_index_[index] == c)).sum(axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNNRegressor(Estimator):
    """Predicts the (weighted) mean target of the k nearest training points."""

    def __init__(self, k=5, weights="uniform", metric="euclidean"):
        self.k = k
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        dist, index = nearest_neighbors(self.X_, np.asarray(X, dtype=float),
                                        self.k, self.metric)
        w = _neighbor_weights(dist, self.weights)
        return (w * self.y_[index]).sum(axis=1) / w.sum(axis=1)
