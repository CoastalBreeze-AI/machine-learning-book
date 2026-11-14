"""Cross-validation, hyperparameter search, and model comparison (Chapter 6).

Splitters yield (train_index, test_index) pairs of integer arrays. X may be
an array or a table (a dict of equal-length columns).
"""
import itertools
import math

import numpy as np

from tinyml.base import clone
from tinyml.metrics import accuracy, mse
from tinyml.preprocessing import _length, _take


class KFold:
    """Splits the rows into n_splits folds; each fold is the test set once."""

    def __init__(self, n_splits=5, shuffle=True, seed=0):
        self.n_splits, self.shuffle, self.seed = n_splits, shuffle, seed

    def split(self, X, y=None, groups=None):
        n = _length(X)
        order = np.arange(n)
        if self.shuffle:
            order = np.random.default_rng(self.seed).permutation(n)
        for test in np.array_split(order, self.n_splits):
            yield np.setdiff1d(order, test), np.sort(test)


class StratifiedKFold(KFold):
    """Like KFold, but each fold keeps the class proportions of y."""

    def split(self, X, y, groups=None):
        rng = np.random.default_rng(self.seed)
        folds = [[] for _ in range(self.n_splits)]
        for label in np.unique(y):
            members = np.flatnonzero(y == label)
            if self.shuffle:
                members = rng.permutation(members)
            for k, part in enumerate(np.array_split(members, self.n_splits)):
                folds[k].extend(part)
        everything = np.arange(len(y))
        for test in folds:
            test = np.sort(np.array(test, dtype=int))
            yield np.setdiff1d(everything, test), test


class GroupKFold:
    """Keeps every row of a group (a customer, a patient) in the same fold,
    so the model is always tested on groups it has never seen."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        unique = np.unique(groups)
        for test_groups in np.array_split(unique, self.n_splits):
            test = np.flatnonzero(np.isin(groups, test_groups))
            yield np.flatnonzero(~np.isin(groups, test_groups)), test


class TimeSeriesSplit:
    """Expanding-window splits for ordered data: train on the past, test on
    the block that follows. Rows must already be in time order."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n = _length(X)
        edges = np.linspace(0, n, self.n_splits + 2).astype(int)
        for k in range(1, self.n_splits + 1):
            yield np.arange(edges[k]), np.arange(edges[k], edges[k + 1])


# Scorers that need a probability or score rather than a hard prediction.
# Any function can override this by setting fn.needs_proba = True or False.
SCORE_BASED = {"roc_auc", "average_precision", "log_loss", "brier_score"}
LOWER_IS_BETTER = {"mse", "mae", "log_loss", "brier_score"}


def _needs_proba(scorer):
    name = getattr(scorer, "__name__", "")
    return getattr(scorer, "needs_proba", name in SCORE_BASED)


def _is_classification(y):
    return np.asarray(y).dtype.kind in "biu"  # integer or Boolean labels


def _resolve(cv, y, scoring):
    if isinstance(cv, int):
        cv = StratifiedKFold(cv) if _is_classification(y) else KFold(cv)
    if scoring is None:
        scoring = {"accuracy": accuracy} if _is_classification(y) else {"mse": mse}
    return cv, scoring


def _score_output(model, X):
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.shape[1] == 2 else proba


def cross_validate(estimator, X, y, cv=5, scoring=None, groups=None,
                   return_train=False):
    """Fits a fresh clone of estimator on each training fold and scores it on
    the held-out fold. Returns {name: array of per-fold scores}.

    scoring maps names to functions f(y_true, y_pred_or_score). A scorer
    receives predict_proba output (column 1 for binary problems) if its name
    is in SCORE_BASED or it has needs_proba = True; otherwise it receives
    predict output. An integer cv means stratified folds for integer labels
    and plain folds otherwise.
    """
    cv, scoring = _resolve(cv, y, scoring)
    results = {name: [] for name in scoring}
    if return_train:
        results.update({f"train_{name}": [] for name in scoring})
    for train, test in cv.split(X, y, groups):
        model = clone(estimator).fit(_take(X, train), y[train])
        parts = [("", test)] + ([("train_", train)] if return_train else [])
        for prefix, rows in parts:
            X_part, y_part = _take(X, rows), y[rows]
            pred = proba = None
            for name, scorer in scoring.items():
                if _needs_proba(scorer):
                    proba = _score_output(model, X_part) if proba is None else proba
                    output = proba
                else:
                    pred = model.predict(X_part) if pred is None else pred
                    output = pred
                results[prefix + name].append(scorer(y_part, output))
    return {name: np.array(values) for name, values in results.items()}


def cross_val_predict(estimator, X, y, cv=5, method="predict", groups=None):
    """Returns an out-of-fold prediction for every row: each row is predicted
    by the model that did not see it. method is "predict" or "predict_proba"
    (which returns P(y=1) for binary problems)."""
    cv, _ = _resolve(cv, y, None)
    out = None
    for train, test in cv.split(X, y, groups):
        model = clone(estimator).fit(_take(X, train), y[train])
        X_test = _take(X, test)
        if method == "predict":
            values = model.predict(X_test)
        else:
            values = _score_output(model, X_test)
        if out is None:
            out = np.zeros((len(y),) + values.shape[1:], dtype=values.dtype)
        out[test] = values
    return out


def set_param(estimator, name, value):
    """Sets a hyperparameter. "model__alpha" means the alpha parameter of the
    pipeline step named "model"."""
    if "__" in name:
        step_name, rest = name.split("__", 1)
        step = dict(estimator.steps)[step_name]
        set_param(step, rest, value)
    else:
        estimator.set_params(**{name: value})
    return estimator


def grid_search(estimator, param_grid, X, y, cv=5, scoring=None, refit=True):
    """Tries every combination in param_grid ({name: list of values}) by
    cross-validation and picks the best mean score.

    scoring is one function. Whether higher is better is read from its name
    (LOWER_IS_BETTER), or from a lower_is_better attribute. Returns a dict
    with best_params, best_score, results, and (if refit) best_estimator,
    refitted on all of X.
    """
    cv, scoring = _resolve(cv, y, None if scoring is None else {"score": scoring})
    scorer = next(iter(scoring.values()))
    lower = getattr(scorer, "lower_is_better",
                    getattr(scorer, "__name__", "") in LOWER_IS_BETTER)
    names = list(param_grid)
    results = []
    for values in itertools.product(*(param_grid[n] for n in names)):
        params = dict(zip(names, values))
        candidate = clone(estimator)
        for name, value in params.items():
            set_param(candidate, name, value)
        scores = next(iter(cross_validate(candidate, X, y, cv, scoring).values()))
        results.append({"params": params, "mean": scores.mean(), "std": scores.std()})
    pick = min if lower else max
    best = pick(results, key=lambda r: r["mean"])
    found = {"best_params": best["params"], "best_score": best["mean"],
             "results": results}
    if refit:
        model = clone(estimator)
        for name, value in best["params"].items():
            set_param(model, name, value)
        found["best_estimator"] = model.fit(X, y)
    return found


def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap confidence interval for metric(y_true, y_pred).

    Resamples test rows with replacement, n_boot times. Returns
    (estimate, low, high) for a 1 - alpha interval. y_pred may be
    predictions or scores, whichever the metric expects.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        stats.append(metric(y_true[rows], y_pred[rows]))
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, y_pred), float(low), float(high)


def paired_bootstrap(y_true, pred_a, pred_b, metric, n_boot=1000, seed=0,
                     alpha=0.05):
    """Bootstrap interval for metric(a) - metric(b) on the same test rows.
    Resampling rows jointly keeps the pairing, which narrows the interval."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        rows = rng.integers(0, n, n)
        diffs.append(metric(y_true[rows], pred_a[rows])
                     - metric(y_true[rows], pred_b[rows]))
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, pred_a) - metric(y_true, pred_b), float(low), float(high)


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on paired predictions from two classifiers.

    b counts rows only A gets right; c counts rows only B gets right. Under
    the null hypothesis of equal accuracy, b ~ Binomial(b + c, 1/2).
    Returns (b, c, two-sided p-value).
    """
    right_a = np.asarray(pred_a) == np.asarray(y_true)
    right_b = np.asarray(pred_b) == np.asarray(y_true)
    b, c = int(np.sum(right_a & ~right_b)), int(np.sum(~right_a & right_b))
    n = b + c
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def _t_two_sided_p(t, df, steps=4000):
    """P(|T| >= |t|) for Student's t with df degrees of freedom.

    Integrates the density by Simpson's rule after substituting
    x = u / (1 - u), which maps [0, inf) onto [0, 1) so that large |t| work.
    """
    u = np.linspace(0.0, abs(t) / (1.0 + abs(t)), steps + 1)
    x = u / (1.0 - u)
    log_c = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
             - 0.5 * math.log(df * math.pi))
    f = np.exp(log_c - (df + 1) / 2 * np.log1p(x ** 2 / df)) / (1.0 - u) ** 2
    h = u[1] - u[0]
    area = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    return float(min(1.0, max(0.0, 1.0 - 2.0 * area)))


def corrected_resampled_ttest(scores_a, scores_b, test_train_ratio):
    """Nadeau and Bengio's corrected t-test for scores from J resampled or
    cross-validated splits. The ordinary paired t-test treats the J
    differences as independent, but overlapping training sets make them
    correlated; the correction inflates the variance by n_test / n_train.
    Returns (t statistic, two-sided p-value) with J - 1 degrees of freedom.
    """
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return (0.0, 1.0) if d.mean() == 0 else (math.copysign(math.inf, d.mean()), 0.0)
    t = d.mean() / math.sqrt((1 / J + test_train_ratio) * var)
    return float(t), _t_two_sided_p(t, J - 1)
