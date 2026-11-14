"""Model-agnostic explanations for models that take tables (Chapter 16).

Both methods need only predict_proba, so they work for any tinyml model or
pipeline. Both work on raw columns, so a categorical column such as
contract is explained as one thing, not as several one-hot columns.
"""
import numpy as np


def neg_log_loss(y_true, p, eps=1e-12):
    """Higher is better, so a drop in score means a loss of skill."""
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def permutation_importance(model, table, y, columns=None, metric=neg_log_loss,
                           n_repeats=5, seed=0):
    """How much the score drops when one column's values are shuffled.

    Shuffling breaks the link between a column and the label while keeping
    the column's distribution. Returns {column: (mean drop, std of drop)},
    sorted from most to least important. Use held-out data.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = metric(y, model.predict_proba(table)[:, 1])
    result = {}
    for name in columns or list(table):
        drops = []
        for _ in range(n_repeats):
            shuffled = dict(table)  # a shallow copy; only one column is replaced
            shuffled[name] = rng.permutation(table[name])
            drops.append(base - metric(y, model.predict_proba(shuffled)[:, 1]))
        result[name] = (float(np.mean(drops)), float(np.std(drops)))
    return dict(sorted(result.items(), key=lambda item: -item[1][0]))


def partial_dependence(model, table, column, grid=None):
    """Average predicted probability when everyone's `column` is set to v.

    For a numeric column the default grid is the 5th to 95th percentiles;
    for a categorical column it is every category. Returns (grid, averages).
    Setting a column without changing correlated ones can create
    combinations that never occur, so read the curve with that in mind.
    """
    values = np.asarray(table[column])
    if grid is None:
        if values.dtype.kind in "fiu":
            grid = np.unique(np.nanpercentile(values, np.linspace(5, 95, 10)))
        else:
            grid = np.unique(values)
    averages = []
    for v in grid:
        changed = dict(table)
        changed[column] = np.full(len(values), v, dtype=values.dtype)
        averages.append(float(model.predict_proba(changed)[:, 1].mean()))
    return np.asarray(grid), np.array(averages)
