"""Drift statistics and reports (Chapter 15).

Each statistic compares a reference sample (usually the training data)
with a current sample (recent inputs). They answer "has the distribution of
this column changed?", not "has the model gotten worse?", which needs labels.
"""
import math

import numpy as np


def _proportions(counts, eps=1e-4):
    p = np.asarray(counts, dtype=float) / max(np.sum(counts), 1)
    return np.clip(p, eps, None)  # an empty bin would make the log infinite


def psi(reference, current, bins=10):
    """Population stability index of a numeric column.

    Bin edges are quantiles of the reference sample, so each reference bin
    holds about the same share. Missing values get a bin of their own.
    PSI = sum over bins of (c - r) * ln(c / r), with r and c the shares.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    ref_ok, cur_ok = reference[~np.isnan(reference)], current[~np.isnan(current)]
    edges = np.unique(np.quantile(ref_ok, np.linspace(0, 1, bins + 1)[1:-1]))
    ref_counts = np.bincount(np.searchsorted(edges, ref_ok, side="right"),
                             minlength=len(edges) + 1)
    cur_counts = np.bincount(np.searchsorted(edges, cur_ok, side="right"),
                             minlength=len(edges) + 1)
    ref_counts = np.append(ref_counts, len(reference) - len(ref_ok))  # missing bin
    cur_counts = np.append(cur_counts, len(current) - len(cur_ok))
    r, c = _proportions(ref_counts), _proportions(cur_counts)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference, current):
    """PSI for a categorical column, with one bin per category."""
    categories = np.union1d(np.unique(reference), np.unique(current))
    r = _proportions([np.sum(np.asarray(reference) == k) for k in categories])
    c = _proportions([np.sum(np.asarray(current) == k) for k in categories])
    return float(np.sum((c - r) * np.log(c / r)))


def ks_statistic(reference, current):
    """Two-sample Kolmogorov-Smirnov statistic D and its asymptotic p-value.

    D is the largest gap between the two empirical CDFs. NaNs are dropped.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(current, dtype=float)
    a, b = np.sort(a[~np.isnan(a)]), np.sort(b[~np.isnan(b)])
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    if lam < 0.3:  # the series converges slowly here, and p is 1 to 5 places
        return d, 1.0
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def chi_square(reference, current):
    """Chi-square test that two samples share one categorical distribution.

    Returns (statistic, approximate p-value). The p-value uses the
    Wilson-Hilferty normal approximation to the chi-square distribution.
    """
    categories = np.union1d(np.unique(reference), np.unique(current))
    table = np.array([[np.sum(np.asarray(s) == k) for k in categories]
                      for s in (reference, current)], dtype=float)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    stat = float(np.sum((table - expected) ** 2 / np.where(expected > 0, expected, 1)))
    dof = max(len(categories) - 1, 1)
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return stat, float(0.5 * math.erfc(z / math.sqrt(2)))


class DriftReport:
    """Compares every column of a current table with a reference table.

    Numeric columns get PSI and a KS test; categorical columns get PSI and a
    chi-square test. The status uses PSI, an effect size, rather than the
    p-value, because with thousands of rows a tiny, harmless change is
    "significant." The thresholds 0.1 and 0.25 are an industry rule of thumb.
    """

    def __init__(self, reference, warn=0.1, alert=0.25, bins=10):
        self.reference = reference
        self.warn, self.alert, self.bins = warn, alert, bins

    def compare(self, current, columns=None):
        self.rows_ = []
        for name in columns or self.reference:
            ref, cur = np.asarray(self.reference[name]), np.asarray(current[name])
            if ref.dtype.kind in "fiu":
                index = psi(ref, cur, self.bins)
                test, (stat, p) = "KS", ks_statistic(ref, cur)
            else:
                index = categorical_psi(ref, cur)
                test, (stat, p) = "chi2", chi_square(ref, cur)
            status = ("ALERT" if index >= self.alert
                      else "warn" if index >= self.warn else "ok")
            self.rows_.append({"column": name, "psi": index, "test": test,
                               "statistic": stat, "p_value": p, "status": status})
        return self

    def drifted(self):
        """Names of the columns whose PSI reached the warning level."""
        return [row["column"] for row in self.rows_ if row["status"] != "ok"]

    def __str__(self):
        lines = [f"{'column':<16}{'PSI':>7}  {'test':<5}{'stat':>9}{'p-value':>10}  status"]
        for r in self.rows_:
            lines.append(f"{r['column']:<16}{r['psi']:>7.3f}  {r['test']:<5}"
                         f"{r['statistic']:>9.3f}{r['p_value']:>10.2g}  {r['status']}")
        return "\n".join(lines)
