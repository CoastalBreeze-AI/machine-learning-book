"""Decision trees for classification and regression (Chapter 8).

Every split is chosen greedily: at each node, try every feature and every
threshold between distinct sorted values, and keep the split that lowers
the node's cost the most. Costs are impurities multiplied by node weight,
so the cost of a split is just the sum of its two children's costs.
"""
import numpy as np

from tinyml.base import Estimator


def gini_cost(counts):
    """Gini impurity times node weight, for each row of class counts."""
    total = counts.sum(axis=1)
    return total - (counts ** 2).sum(axis=1) / np.maximum(total, 1e-300)


def entropy_cost(counts):
    """Entropy (in bits) times node weight; 0 log 0 is treated as 0."""
    total = counts.sum(axis=1, keepdims=True)
    share = np.maximum(counts, 1e-300) / np.maximum(total, 1e-300)
    return -(counts * np.log2(share)).sum(axis=1)


def squared_error_cost(stats):
    """Sum of squared errors around the mean, from columns [w, w*y, w*y^2]."""
    w, wy, wy2 = stats[:, 0], stats[:, 1], stats[:, 2]
    return np.maximum(wy2 - wy ** 2 / np.maximum(w, 1e-300), 0.0)


def best_split(x, stats, cost, min_leaf):
    """Finds the best threshold on one feature, testing all of them at once.

    x holds the feature for the rows in this node, and stats holds one row
    of additive statistics per sample (class counts, or [w, w*y, w*y^2]).
    After sorting, a cumulative sum gives the statistics of every possible
    left child, and subtraction gives every right child. Returns
    (cost of the two children, threshold), or None if no split is allowed.
    """
    order = np.argsort(x, kind="stable")
    xs = x[order]
    left = np.cumsum(stats[order], axis=0)[:-1]  # left child = first i+1 rows
    right = left[-1] + stats[order[-1]] - left
    n_left = np.arange(1, len(x))
    valid = ((xs[1:] > xs[:-1])  # never split between equal values
             & (n_left >= min_leaf) & (len(x) - n_left >= min_leaf))
    if not valid.any():
        return None
    total = np.where(valid, cost(left) + cost(right), np.inf)
    i = np.argmin(total)
    return total[i], (xs[i] + xs[i + 1]) / 2


def _n_features(max_features, d):
    """Turns max_features (None, "sqrt", "log2", a fraction, or an int) into a count."""
    if max_features is None:
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    if isinstance(max_features, float):
        return max(1, int(max_features * d))
    return min(int(max_features), d)


class _BaseTree(Estimator):
    """Grows a binary tree. Subclasses say how to summarize the targets."""

    def _grow(self, X, stats):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        n_try = _n_features(self.max_features, d)
        rng = np.random.default_rng(self.seed)
        max_depth = np.inf if self.max_depth is None else self.max_depth
        self.feature_, self.threshold_, self.left_, self.right_ = [], [], [], []
        self.value_, self.n_node_samples_ = [], []
        importance = np.zeros(d)

        def build(rows, depth):
            node = len(self.feature_)
            total = stats[rows].sum(axis=0)
            node_cost = self._cost(total[None, :])[0]
            for attr, v in [(self.feature_, -1), (self.threshold_, np.nan),
                            (self.left_, -1), (self.right_, -1),
                            (self.value_, self._leaf_value(total)),
                            (self.n_node_samples_, len(rows))]:
                attr.append(v)
            if (depth >= max_depth or len(rows) < 2 * self.min_samples_leaf
                    or node_cost <= 1e-12):  # stop: too deep, too small, or pure
                return node
            features = (rng.choice(d, n_try, replace=False) if n_try < d
                        else range(d))
            best = (np.inf, -1, 0.0)
            for f in features:
                found = best_split(X[rows, f], stats[rows], self._cost,
                                   self.min_samples_leaf)
                if found is not None and found[0] < best[0]:
                    best = (found[0], f, found[1])
            child_cost, f, threshold = best
            if f < 0:
                return node  # no allowed split on the features tried
            importance[f] += node_cost - child_cost
            goes_left = X[rows, f] <= threshold
            self.feature_[node], self.threshold_[node] = f, threshold
            self.left_[node] = build(rows[goes_left], depth + 1)
            self.right_[node] = build(rows[~goes_left], depth + 1)
            return node

        build(np.arange(n), 0)
        for name in ["feature_", "threshold_", "left_", "right_", "value_",
                     "n_node_samples_"]:
            setattr(self, name, np.array(getattr(self, name)))
        self.feature_importances_ = importance / max(importance.sum(), 1e-300)
        return self

    def apply(self, X):
        """Returns the index of the leaf that each row of X falls into."""
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            rows = np.flatnonzero(self.left_[node] != -1)  # rows not yet at a leaf
            if len(rows) == 0:
                return node
            at = node[rows]
            goes_left = X[rows, self.feature_[at]] <= self.threshold_[at]
            node[rows] = np.where(goes_left, self.left_[at], self.right_[at])


class DecisionTreeClassifier(_BaseTree):
    """A CART-style classification tree with Gini or entropy impurity."""

    def __init__(self, max_depth=None, min_samples_leaf=1, criterion="gini",
                 max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        self.classes_, codes = np.unique(y, return_inverse=True)
        w = np.ones(len(codes)) if sample_weight is None else np.asarray(sample_weight)
        stats = np.zeros((len(codes), len(self.classes_)))
        stats[np.arange(len(codes)), codes] = w  # weighted one-hot class counts
        self._cost = {"gini": gini_cost, "entropy": entropy_cost}[self.criterion]
        return self._grow(X, stats)

    def _leaf_value(self, counts):
        return counts / counts.sum()  # class proportions in the leaf

    def _describe(self, node, decimals):
        p = self.value_[node]
        shown = f"P({self.classes_[1]})={p[1]:.{decimals}f}" if len(p) == 2 else \
            "P=" + np.array2string(p, precision=decimals)
        return f"leaf: n={self.n_node_samples_[node]}, {shown}"

    def predict_proba(self, X):
        return self.value_[self.apply(X)]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DecisionTreeRegressor(_BaseTree):
    """A regression tree that minimizes squared error; leaves predict a mean."""

    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        stats = np.column_stack([w, w * y, w * y ** 2])
        self._cost = squared_error_cost
        return self._grow(X, stats)

    def _leaf_value(self, stats):
        return stats[1] / stats[0]  # weighted mean of y in the leaf

    def _describe(self, node, decimals):
        return f"leaf: n={self.n_node_samples_[node]}, value={self.value_[node]:.{decimals}f}"

    def predict(self, X):
        return self.value_[self.apply(X)]


def export_text(tree, feature_names=None, decimals=2):
    """Returns a fitted tree as indented text, one line per split or leaf."""
    d = len(tree.feature_importances_)
    names = feature_names or [f"x{j}" for j in range(d)]
    lines = []

    def walk(node, depth):
        pad = "|   " * depth + "|--- "
        if tree.left_[node] == -1:
            lines.append(pad + tree._describe(node, decimals))
            return
        name, t = names[tree.feature_[node]], tree.threshold_[node]
        lines.append(f"{pad}{name} <= {t:.{decimals}f}")
        walk(tree.left_[node], depth + 1)
        lines.append(f"{pad}{name} >  {t:.{decimals}f}")
        walk(tree.right_[node], depth + 1)

    walk(0, 0)
    return "\n".join(lines)
