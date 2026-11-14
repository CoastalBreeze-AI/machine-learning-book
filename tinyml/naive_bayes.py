"""Naive Bayes classifiers (Chapter 9).

Each model learns p(y) and p(x | y) under the assumption that features are
independent given the class, then applies Bayes' rule. All the arithmetic is
done with logarithms, because a product of many small densities underflows.
"""
import numpy as np

from tinyml.base import Estimator


def log_normalize_rows(scores):
    """Subtracts each row's log-sum-exp, so that exp(row) sums to 1."""
    top = scores.max(axis=1, keepdims=True)
    return scores - top - np.log(np.exp(scores - top).sum(axis=1, keepdims=True))


class NaiveBayes(Estimator):
    """Shared prediction code. A subclass sets classes_ and class_log_prior_
    and supplies _feature_log_likelihood(X), an (n, k) array whose entry
    [i, c] is the sum over features j of log p(x_ij | y = c)."""

    def _joint_log_likelihood(self, X):
        return self.class_log_prior_ + self._feature_log_likelihood(X)

    def predict_log_proba(self, X):
        return log_normalize_rows(self._joint_log_likelihood(X))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]


def _class_prior(y):
    classes = np.unique(y)
    return classes, np.log([np.mean(y == c) for c in classes])


class GaussianNB(NaiveBayes):
    """Each feature is normal within each class, independent of the others.

    var_smoothing adds that fraction of the largest feature variance to every
    variance, so a feature that is constant within a class can't produce a
    zero variance. NaNs are skipped when fitting and predicting: under the
    independence assumption, a missing feature simply drops out of the product.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, self.class_log_prior_ = _class_prior(y)
        self.theta_ = np.array([np.nanmean(X[y == c], axis=0) for c in self.classes_])
        var = np.array([np.nanvar(X[y == c], axis=0) for c in self.classes_])
        self.epsilon_ = self.var_smoothing * np.nanvar(X, axis=0).max()
        self.var_ = var + self.epsilon_
        return self

    def _feature_log_likelihood(self, X):
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.theta_[None, :, :]          # shape (n, k, d)
        log_pdf = -0.5 * (np.log(2 * np.pi * self.var_) + diff ** 2 / self.var_)
        return np.nansum(log_pdf, axis=2)                       # NaN terms count as 0


class CategoricalNB(NaiveBayes):
    """Each feature is a categorical variable within each class.

    X is a 2-D array of category values (strings or integers), one column per
    feature. Probabilities are smoothed counts:
        p(x_j = v | y = c) = (count(v, c) + alpha) / (count(c) + alpha * K_j),
    where K_j is the number of categories of feature j seen in training.
    A category never seen in training contributes nothing, like a missing value.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim == 1:
            X = X[:, None]
        self.classes_, self.class_log_prior_ = _class_prior(y)
        self.categories_, self.feature_log_prob_ = [], []
        for column in X.T:
            cats = np.unique(column)
            counts = np.array([[np.sum((column == v) & (y == c)) for v in cats]
                               for c in self.classes_], dtype=float)
            smoothed = counts + self.alpha
            self.categories_.append(cats)
            self.feature_log_prob_.append(
                np.log(smoothed / smoothed.sum(axis=1, keepdims=True)))
        return self

    def _feature_log_likelihood(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        total = np.zeros((len(X), len(self.classes_)))
        for column, cats, log_prob in zip(X.T, self.categories_, self.feature_log_prob_):
            index = np.searchsorted(cats, column).clip(0, len(cats) - 1)
            known = cats[index] == column
            total[known] += log_prob[:, index[known]].T
        return total


class MixedNB(NaiveBayes):
    """Naive Bayes for a table: Gaussian for numeric columns, categorical for
    the rest. Because features are independent given the class, the two
    models' log-likelihoods simply add, with the class prior counted once."""

    def __init__(self, numeric, categorical, var_smoothing=1e-9, alpha=1.0):
        self.numeric = numeric
        self.categorical = categorical
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def _split(self, table):
        num = np.column_stack([np.asarray(table[c], dtype=float) for c in self.numeric])
        cat = np.column_stack([table[c] for c in self.categorical])
        return num, cat

    def fit(self, table, y):
        num, cat = self._split(table)
        self.gaussian_ = GaussianNB(self.var_smoothing).fit(num, y)
        self.categorical_ = CategoricalNB(self.alpha).fit(cat, y)
        self.classes_ = self.gaussian_.classes_
        self.class_log_prior_ = self.gaussian_.class_log_prior_
        return self

    def _feature_log_likelihood(self, table):
        num, cat = self._split(table)
        return (self.gaussian_._feature_log_likelihood(num)
                + self.categorical_._feature_log_likelihood(cat))
