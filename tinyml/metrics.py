"""Metrics for regression and classification (Chapter 6).

Every function takes (y_true, y_pred_or_score). Classification metrics that
need hard labels take predictions; ranking and probability metrics take a
score, which for a binary problem is the predicted probability of class 1.
"""
import numpy as np


def mse(y_true, y_pred):
    """Mean squared error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2(y_true, y_pred):
    """Coefficient of determination: 1 - SSE / SST. It is negative when the
    model does worse than always predicting the mean of y_true."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / sst)


def confusion_matrix(y_true, y_pred, labels=None):
    """Counts with rows = true class, columns = predicted class.

    For labels [0, 1] this is [[TN, FP], [FN, TP]].
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    return matrix


def accuracy(y_true, y_pred):
    """Fraction of predictions that are exactly right."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred, positive=1):
    """Of the cases predicted positive, the fraction that are positive."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    predicted = y_pred == positive
    if not predicted.any():
        return 0.0  # a common convention; the ratio is undefined
    return float(np.mean(y_true[predicted] == positive))


def recall(y_true, y_pred, positive=1):
    """Of the cases that are positive, the fraction predicted positive."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    actual = y_true == positive
    if not actual.any():
        return 0.0
    return float(np.mean(y_pred[actual] == positive))


def f1(y_true, y_pred, positive=1):
    """Harmonic mean of precision and recall."""
    p, r = precision(y_true, y_pred, positive), recall(y_true, y_pred, positive)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def roc_curve(y_true, score):
    """False and true positive rates at every distinct threshold.

    Returns (fpr, tpr, thresholds). The case is predicted positive when
    score >= threshold. The curve starts at (0, 0) with threshold +inf.
    """
    y_true, score = np.asarray(y_true), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    score, y_sorted = score[order], y_true[order] == 1
    # Keep the last position of each run of tied scores, so ties move together.
    last = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    tpr = np.r_[0.0, tp / max(y_sorted.sum(), 1)]
    fpr = np.r_[0.0, fp / max((~y_sorted).sum(), 1)]
    return fpr, tpr, np.r_[np.inf, score[last]]


def roc_auc(y_true, score):
    """Area under the ROC curve, by the trapezoidal rule. It equals the
    probability that a random positive outscores a random negative, with
    ties counted as one half."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def precision_recall_curve(y_true, score):
    """Precision and recall at every distinct threshold, highest first.

    Returns (precision, recall, thresholds), one entry per threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, score)
    y_true = np.asarray(y_true)
    n_pos, n_neg = np.sum(y_true == 1), np.sum(y_true != 1)
    tp, fp = tpr[1:] * n_pos, fpr[1:] * n_neg
    return tp / (tp + fp), tpr[1:], thresholds[1:]


def average_precision(y_true, score):
    """Area under the precision-recall curve, as a step function of recall."""
    prec, rec, _ = precision_recall_curve(y_true, score)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _proba_matrix(proba):
    """Turns a vector of P(y=1) into an (n, 2) matrix; leaves matrices alone."""
    proba = np.asarray(proba, dtype=float)
    return np.column_stack([1 - proba, proba]) if proba.ndim == 1 else proba


def log_loss(y_true, proba, eps=1e-15):
    """Mean negative log-likelihood (cross-entropy) of the true labels.

    proba is P(y=1) for a binary problem, or an (n, k) matrix whose columns
    follow the sorted class labels, in which case y_true holds 0, ..., k-1.
    """
    proba = np.clip(_proba_matrix(proba), eps, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true])))


def brier_score(y_true, proba):
    """Mean squared difference between the predicted probability and the
    0/1 outcome. Lower is better; always predicting 0.5 scores 0.25."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]
    return float(np.mean((proba - np.asarray(y_true)) ** 2))


def calibration_curve(y_true, proba, n_bins=10):
    """Groups predictions into equal-width probability bins.

    Returns (mean_predicted, fraction_positive, counts) for each nonempty
    bin. A calibrated model has mean_predicted close to fraction_positive.
    """
    y_true, proba = np.asarray(y_true, dtype=float), np.asarray(proba, dtype=float)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    used = [b for b in range(n_bins) if np.any(bins == b)]
    mean_predicted = np.array([proba[bins == b].mean() for b in used])
    fraction_positive = np.array([y_true[bins == b].mean() for b in used])
    counts = np.array([np.sum(bins == b) for b in used])
    return mean_predicted, fraction_positive, counts
