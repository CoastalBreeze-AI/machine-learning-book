"""Splitting and preprocessing (Chapter 3).

A table is a dict that maps column names to equal-length NumPy arrays, as
returned by make_churn. Models need a numeric matrix, so TableEncoder turns a
table into one, learning everything it needs from the training data only.
"""
import numpy as np

from tinyml.base import Transformer


def _take(data, index):
    if isinstance(data, dict):
        return {name: column[index] for name, column in data.items()}
    return data[index]


def _length(data):
    return len(next(iter(data.values()))) if isinstance(data, dict) else len(data)


def train_test_split(X, y, test_size=0.2, seed=0, stratify=False):
    """Randomly splits X (an array or a table) and y into train and test parts.

    With stratify=True, each class keeps the same proportion in both parts.
    """
    rng = np.random.default_rng(seed)
    n = _length(X)
    if stratify:
        test = []
        for label in np.unique(y):
            members = rng.permutation(np.flatnonzero(y == label))
            test.extend(members[:round(test_size * len(members))])
        test = np.array(sorted(test), dtype=int)
    else:
        test = np.sort(rng.permutation(n)[:round(test_size * n)])
    train = np.setdiff1d(np.arange(n), test)
    return _take(X, train), _take(X, test), y[train], y[test]


class SimpleImputer(Transformer):
    """Replaces NaNs in each column with that column's training mean or median."""

    def __init__(self, strategy="median"):
        self.strategy = strategy

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        stat = np.nanmedian if self.strategy == "median" else np.nanmean
        with np.errstate(all="ignore"):
            self.fill_ = np.array([stat(col) if not np.isnan(col).all() else 0.0
                                   for col in X.T])
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        rows, cols = np.nonzero(np.isnan(X))
        X[rows, cols] = self.fill_[cols]
        return X


class StandardScaler(Transformer):
    """Rescales each column to mean 0 and standard deviation 1, using training statistics."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)  # leave constant columns alone
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_


class OneHotEncoder(Transformer):
    """Turns each categorical column into one 0/1 column per category seen in training.

    A category that never appeared in training is encoded as all zeros.
    """

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        self.categories_ = [np.unique(col) for col in X.T]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        parts = [(col[:, None] == cats[None, :]).astype(float)
                 for col, cats in zip(X.T, self.categories_)]
        return np.hstack(parts)


class TableEncoder(Transformer):
    """Converts a table to a numeric matrix: impute and scale numeric columns,
    one-hot encode categorical ones. Columns not listed are dropped.
    """

    def __init__(self, numeric, categorical, missing_indicators=False):
        self.numeric = numeric
        self.categorical = categorical
        self.missing_indicators = missing_indicators

    def _numeric(self, table):
        return np.column_stack([np.asarray(table[c], dtype=float) for c in self.numeric])

    def fit(self, table, y=None):
        names = []
        if self.numeric:
            raw = self._numeric(table)
            self.imputer_ = SimpleImputer().fit(raw)
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(raw))
            names += list(self.numeric)
            if self.missing_indicators:
                self.flagged_ = [i for i, col in enumerate(raw.T) if np.isnan(col).any()]
                names += [f"{self.numeric[i]}_missing" for i in self.flagged_]
        if self.categorical:
            cats = np.column_stack([table[c] for c in self.categorical])
            self.onehot_ = OneHotEncoder().fit(cats)
            for column, values in zip(self.categorical, self.onehot_.categories_):
                names += [f"{column}={value}" for value in values]
        self.feature_names_ = names
        return self

    def transform(self, table):
        parts = []
        if self.numeric:
            raw = self._numeric(table)
            parts.append(self.scaler_.transform(self.imputer_.transform(raw)))
            if self.missing_indicators:
                parts.append(np.isnan(raw[:, self.flagged_]).astype(float))
        if self.categorical:
            cats = np.column_stack([table[c] for c in self.categorical])
            parts.append(self.onehot_.transform(cats))
        return np.hstack(parts)
