"""Ensembles of trees: bagging, random forests, and boosting (Chapter 8)."""
import numpy as np

from tinyml.base import Estimator
from tinyml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(Estimator):
    """Averages the class probabilities of trees grown on bootstrap samples,
    each split choosing among a random subset of max_features features.

    With max_features=None, every split sees every feature, which is bagging.
    """

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, criterion="gini", seed=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        n, k = len(codes), len(self.classes_)
        rng = np.random.default_rng(self.seed)
        self.estimators_ = []
        oob_votes = np.zeros((n, k))
        for _ in range(self.n_estimators):
            rows = rng.integers(0, n, n)  # a bootstrap sample, with replacement
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf,
                                          self.criterion, self.max_features,
                                          seed=int(rng.integers(2 ** 31)))
            tree.fit(X[rows], codes[rows])
            self.estimators_.append(tree)
            out = np.ones(n, dtype=bool)
            out[rows] = False  # about 37% of rows are out of bag for each tree
            oob_votes[np.ix_(out, tree.classes_)] += tree.predict_proba(X[out])
        voted = oob_votes.sum(axis=1) > 0
        self.oob_proba_ = oob_votes[voted] / oob_votes[voted].sum(axis=1, keepdims=True)
        self.oob_score_ = float(np.mean(np.argmax(self.oob_proba_, axis=1) == codes[voted]))
        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.estimators_], axis=0)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        proba = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            proba[:, tree.classes_] += tree.predict_proba(X)  # a tree may miss a class
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # 1 / (1 + e^-z), without overflow


class AdaBoostClassifier(Estimator):
    """Discrete AdaBoost for two classes, with shallow trees as weak learners."""

    def __init__(self, n_estimators=50, max_depth=1, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        signs = np.where(y == self.classes_[1], 1.0, -1.0)
        w = np.full(len(signs), 1.0 / len(signs))
        self.estimators_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(self.max_depth, seed=self.seed)
            tree.fit(X, signs, sample_weight=w)
            h = tree.predict(X)
            error = np.sum(w[h != signs]) / np.sum(w)
            if error >= 0.5:  # no better than chance on the weighted data
                break
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-12))
            self.estimators_.append(tree)
            self.alphas_.append(alpha)
            if error == 0:  # a perfect learner: reweighting would change nothing
                break
            w = w * np.exp(-alpha * signs * h)  # up-weight the mistakes
            w /= w.sum()
        return self

    def decision_function(self, X):
        F = np.zeros(len(X))  # with no learners, every score is 0
        for alpha, tree in zip(self.alphas_, self.estimators_):
            F += alpha * tree.predict(X)
        return F

    def predict_proba(self, X):
        p = _sigmoid(2 * self.decision_function(X))  # F estimates half the log-odds
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class GradientBoostingClassifier(Estimator):
    """Binary gradient boosting with log-loss and regression trees.

    Each stage fits a regression tree to the negative gradient of the loss,
    y - p, then sets every leaf to a Newton step, sum(y - p) / sum(p(1 - p)).
    The model's output F(x) is the log-odds of class 1.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, min_samples_leaf=1, seed=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        y01 = (y == self.classes_[1]).astype(float)
        rng = np.random.default_rng(self.seed)
        n = len(y01)
        self.init_ = np.log(y01.mean() / (1 - y01.mean()))  # the best constant
        F = np.full(n, self.init_)
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            p = _sigmoid(F)
            residual = y01 - p  # the negative gradient of log-loss with respect to F
            rows = (rng.choice(n, round(self.subsample * n), replace=False)
                    if self.subsample < 1 else np.arange(n))
            tree = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf,
                                         seed=int(rng.integers(2 ** 31)))
            tree.fit(X[rows], residual[rows])
            leaf = tree.apply(X[rows])
            numer = np.bincount(leaf, residual[rows], minlength=len(tree.value_))
            denom = np.bincount(leaf, (p * (1 - p))[rows], minlength=len(tree.value_))
            is_leaf = tree.left_ == -1
            tree.value_[is_leaf] = numer[is_leaf] / np.maximum(denom[is_leaf], 1e-12)
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean(np.logaddexp(0, F) - y01 * F)))
        return self

    def staged_decision_function(self, X):
        """Yields F(x) after each stage, for choosing n_estimators on validation data."""
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
            yield F

    def decision_function(self, X):
        F = np.full(len(X), self.init_)
        for F in self.staged_decision_function(X):
            pass
        return F

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
