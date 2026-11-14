"""Group fairness metrics (Chapter 16).

Every function takes true labels, hard predictions or scores, and a group
label for each row. Which group gap matters, and in which direction, is a
decision about the application that no metric can make for you.
"""
import numpy as np


def _rate(mask):
    return float(np.mean(mask)) if np.size(mask) else float("nan")


def group_rates(y_true, y_pred, group):
    """Per-group counts and rates: selection rate, TPR, FPR, and precision."""
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    rates = {}
    for g in np.unique(group):
        t, p = y_true[group == g], y_pred[group == g]
        rates[str(g)] = {
            "n": int(len(t)),
            "base_rate": _rate(t == 1),
            "selection_rate": _rate(p == 1),
            "tpr": _rate(p[t == 1] == 1),
            "fpr": _rate(p[t == 0] == 1),
            "precision": _rate(t[p == 1] == 1),
        }
    return rates


def selection_rate(y_pred, group):
    """The share of each group that the model selects (predicts as 1)."""
    y_pred, group = np.asarray(y_pred), np.asarray(group)
    return {str(g): _rate(y_pred[group == g] == 1) for g in np.unique(group)}


def demographic_parity_difference(y_pred, group):
    """Largest gap in selection rate between any two groups."""
    rates = list(selection_rate(y_pred, group).values())
    return max(rates) - min(rates)


def disparate_impact_ratio(y_pred, group):
    """Lowest selection rate divided by the highest (the four-fifths rule
    treats a ratio below 0.8 as a sign of possible adverse impact)."""
    rates = list(selection_rate(y_pred, group).values())
    return min(rates) / max(rates) if max(rates) > 0 else float("nan")


def equalized_odds_difference(y_true, y_pred, group):
    """The larger of the TPR gap and the FPR gap across groups."""
    rates = group_rates(y_true, y_pred, group).values()
    tpr = [r["tpr"] for r in rates]
    fpr = [r["fpr"] for r in rates]
    return max(max(tpr) - min(tpr), max(fpr) - min(fpr))


def threshold_sweep(y_true, score, group, thresholds=None):
    """Selection rate, TPR, and FPR for each group at each threshold.

    Returns {group: array of shape (len(thresholds), 4)} whose columns are
    threshold, selection rate, TPR, and FPR.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    y_true, score, group = np.asarray(y_true), np.asarray(score), np.asarray(group)
    sweep = {}
    for g in np.unique(group):
        t, s = y_true[group == g], score[group == g]
        rows = [(th, _rate(s >= th), _rate(s[t == 1] >= th), _rate(s[t == 0] >= th))
                for th in thresholds]
        sweep[str(g)] = np.array(rows)
    return sweep


def thresholds_for_tpr(y_true, score, group, target_tpr):
    """Per-group thresholds that give each group (about) the same TPR.

    For each group, picks the highest threshold whose TPR reaches the target,
    which equalizes opportunity (Hardt et al.) but not the false positive rate.
    """
    y_true, score, group = np.asarray(y_true), np.asarray(score), np.asarray(group)
    chosen = {}
    for g in np.unique(group):
        positives = np.sort(score[(group == g) & (y_true == 1)])[::-1]
        k = max(int(np.ceil(target_tpr * len(positives))), 1)
        chosen[str(g)] = float(positives[k - 1])  # the k-th highest positive score
    return chosen


def group_calibration(y_true, score, group, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Mean score and observed rate in each score bin, for each group.

    A score is calibrated within groups if, in every group, people given a
    score near s turn out positive about a fraction s of the time.
    """
    y_true, score, group = np.asarray(y_true), np.asarray(score), np.asarray(group)
    table = {}
    for g in np.unique(group):
        t, s = y_true[group == g], score[group == g]
        index = np.clip(np.digitize(s, bins[1:-1]), 0, len(bins) - 2)
        table[str(g)] = [(float(s[index == b].mean()), float(t[index == b].mean()),
                          int(np.sum(index == b)))
                         for b in range(len(bins) - 1) if np.any(index == b)]
    return table
