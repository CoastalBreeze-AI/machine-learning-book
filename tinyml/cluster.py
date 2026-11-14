"""Clustering: k-means, Gaussian mixtures, and measures of cluster quality (Chapter 10)."""
import numpy as np

from tinyml.base import Estimator


def squared_distances(X, C):
    """Returns the (n, k) squared Euclidean distances from rows of X to rows of C."""
    d2 = (X ** 2).sum(axis=1)[:, None] - 2 * X @ C.T + (C ** 2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)                 # rounding can make tiny values negative


def kmeans_plus_plus(X, k, rng):
    """Chooses k starting centers, each new one with probability proportional
    to its squared distance from the nearest center chosen so far."""
    centers = [X[rng.integers(len(X))]]
    d2 = squared_distances(X, centers[0][None, :])[:, 0]
    for _ in range(1, k):
        total = d2.sum()
        p = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
        centers.append(X[rng.choice(len(X), p=p)])
        d2 = np.minimum(d2, squared_distances(X, centers[-1][None, :])[:, 0])
    return np.array(centers)


class KMeans(Estimator):
    """Lloyd's algorithm with k-means++ starts. The best of n_init runs, by
    inertia (the within-cluster sum of squared distances), is kept."""

    def __init__(self, n_clusters=8, n_init=10, n_iter=300, tol=1e-8, seed=0):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def _lloyd(self, X, centers):
        for iteration in range(1, self.n_iter + 1):
            labels = squared_distances(X, centers).argmin(axis=1)      # assign
            new = centers.copy()
            for j in range(self.n_clusters):                            # update
                members = X[labels == j]
                if len(members):               # an empty cluster keeps its center
                    new[j] = members.mean(axis=0)
            shift = np.sum((new - centers) ** 2)
            centers = new
            if shift <= self.tol:
                break
        d2 = squared_distances(X, centers)
        labels = d2.argmin(axis=1)
        return centers, labels, d2[np.arange(len(X)), labels].sum(), iteration

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            run = self._lloyd(X, kmeans_plus_plus(X, self.n_clusters, rng))
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        return squared_distances(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)

    def transform(self, X):
        """Distances from each row to each center."""
        return np.sqrt(squared_distances(np.asarray(X, dtype=float), self.cluster_centers_))


def silhouette_samples(X, labels):
    """s_i = (b_i - a_i) / max(a_i, b_i), where a_i is the mean distance from
    point i to the rest of its cluster and b_i the mean distance to the
    nearest other cluster. Points alone in their cluster get 0."""
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    D = np.sqrt(squared_distances(X, X))
    np.fill_diagonal(D, 0.0)
    clusters, own, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    sums = np.column_stack([D[:, labels == c].sum(axis=1) for c in clusters])
    rows = np.arange(len(X))
    a = sums[rows, own] / np.maximum(sizes[own] - 1, 1)
    means = sums / sizes
    means[rows, own] = np.inf
    b = means.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return np.where(sizes[own] > 1, s, 0.0)


def silhouette_score(X, labels, sample_size=None, seed=0):
    """Mean silhouette, optionally on a random subsample to save memory."""
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if sample_size is not None and sample_size < len(X):
        keep = np.random.default_rng(seed).choice(len(X), sample_size, replace=False)
        X, labels = X[keep], labels[keep]
    return float(np.mean(silhouette_samples(X, labels)))


def adjusted_rand_index(labels_true, labels_pred):
    """Agreement between two partitions, corrected for chance: 1 means
    identical (up to renaming clusters), and about 0 means random."""
    _, t = np.unique(labels_true, return_inverse=True)
    _, p = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((t.max() + 1, p.max() + 1))
    np.add.at(table, (t, p), 1)

    def pairs(counts):                   # number of pairs within each group, summed
        return np.sum(counts * (counts - 1) / 2)

    both, rows, cols = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * cols / pairs(np.array([len(t)]))
    maximum = (rows + cols) / 2
    return 1.0 if maximum == expected else float((both - expected) / (maximum - expected))


def logsumexp(a, axis):
    """log(sum(exp(a))) along an axis, without overflow or underflow."""
    top = a.max(axis=axis, keepdims=True)
    return (top + np.log(np.exp(a - top).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_gaussians(X, means, covariances, covariance_type):
    """Returns the (n, k) array of log N(x_i | means[j], covariances[j])."""
    n, d = X.shape
    out = np.empty((n, len(means)))
    for j, (mean, cov) in enumerate(zip(means, covariances)):
        diff = X - mean
        if covariance_type == "full":
            L = np.linalg.cholesky(cov)                     # cov = L L^T
            z = np.linalg.solve(L, diff.T)                  # whitened differences
            mahalanobis = np.sum(z ** 2, axis=0)
            log_det = 2 * np.sum(np.log(np.diag(L)))
        else:                                               # cov holds d variances
            mahalanobis = np.sum(diff ** 2 / cov, axis=1)
            log_det = np.sum(np.log(cov))
        out[:, j] = -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)
    return out


def m_step(X, resp, covariance_type, reg_covar):
    """Weighted maximum likelihood: each point counts toward component j in
    proportion to its responsibility resp[i, j]. Returns weights, means, covs."""
    n, d = X.shape
    counts = resp.sum(axis=0) + 1e-10                       # soft cluster sizes
    means = resp.T @ X / counts[:, None]
    covs = []
    for j in range(resp.shape[1]):
        diff = X - means[j]
        if covariance_type == "full":
            cov = (resp[:, j, None] * diff).T @ diff / counts[j]
            covs.append(cov + reg_covar * np.eye(d))
        else:
            covs.append(resp[:, j] @ diff ** 2 / counts[j] + reg_covar)
    return counts / n, means, np.array(covs)


class GaussianMixture(Estimator):
    """A mixture of Gaussians fitted by expectation-maximization (EM).

    covariance_type is "full" (any ellipse) or "diag" (axis-aligned ellipses).
    reg_covar is added to every variance to keep covariances invertible.
    log_likelihood_history_ holds the mean log-likelihood per point after
    initialization and after each EM iteration; EM never lets it decrease.
    """

    def __init__(self, n_components=2, covariance_type="full", n_iter=100,
                 tol=1e-6, reg_covar=1e-6, seed=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def _update(self, X, resp):
        self.weights_, self.means_, self.covariances_ = m_step(
            X, resp, self.covariance_type, self.reg_covar)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        start = KMeans(self.n_components, n_init=1, seed=self.seed).fit(X).labels_
        self._update(X, np.eye(self.n_components)[start])  # start from k-means
        history = [self.score(X)]
        for _ in range(self.n_iter):
            self._update(X, self.predict_proba(X))           # E-step, then M-step
            history.append(self.score(X))
            if history[-1] - history[-2] < self.tol:
                break
        self.log_likelihood_history_ = history
        self.n_iter_ = len(history) - 1
        return self

    def _log_joint(self, X):
        X = np.asarray(X, dtype=float)
        return np.log(self.weights_) + log_gaussians(
            X, self.means_, self.covariances_, self.covariance_type)

    def score_samples(self, X):
        """log p(x) for each row, a log-sum-exp over the components."""
        return logsumexp(self._log_joint(X), axis=1)

    def score(self, X):
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X):
        """Responsibilities: p(component j | x) for each row."""
        log_joint = self._log_joint(X)
        return np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

    def predict(self, X):
        return self._log_joint(X).argmax(axis=1)

    def bic(self, X):
        """Bayesian information criterion; lower is better."""
        n, d = np.shape(X)
        per_cov = d * (d + 1) / 2 if self.covariance_type == "full" else d
        n_params = (self.n_components - 1) + self.n_components * (d + per_cov)
        return -2 * self.score(X) * n + n_params * np.log(n)
