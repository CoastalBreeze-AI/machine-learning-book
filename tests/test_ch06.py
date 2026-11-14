"""Tests for tinyml/metrics.py and tinyml/model_selection.py (Chapter 6).

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch06
"""
import itertools
import math
import unittest

import numpy as np

from tinyml.base import Estimator, Pipeline
from tinyml.data import make_churn
from tinyml.linear import LogisticRegression, Ridge
from tinyml.metrics import (accuracy, average_precision, brier_score,
                            calibration_curve, confusion_matrix, f1, log_loss,
                            mae, mse, precision, precision_recall_curve, r2,
                            recall, roc_auc, roc_curve)
from tinyml.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                    TimeSeriesSplit, _t_two_sided_p,
                                    bootstrap_ci, corrected_resampled_ttest,
                                    cross_val_predict, cross_validate,
                                    grid_search, mcnemar, paired_bootstrap,
                                    set_param)
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]


def pairwise_auc(y, s):
    """AUC by brute force over all positive-negative pairs."""
    pos, neg = s[y == 1], s[y == 0]
    wins = sum((p > q) + 0.5 * (p == q) for p, q in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


class TestMetrics(unittest.TestCase):
    def test_regression(self):
        y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 6.0])
        self.assertAlmostEqual(mse(y, p), 1.0)
        self.assertAlmostEqual(mae(y, p), 0.5)
        self.assertAlmostEqual(r2(y, y), 1.0)
        self.assertAlmostEqual(r2(y, np.full(4, y.mean())), 0.0)

    def test_confusion_and_rates(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        p = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(confusion_matrix(y, p), [[3, 1], [1, 3]])
        self.assertAlmostEqual(accuracy(y, p), 0.75)
        self.assertAlmostEqual(precision(y, p), 0.75)
        self.assertAlmostEqual(recall(y, p), 0.75)
        self.assertAlmostEqual(f1(y, p), 0.75)
        self.assertEqual(precision(y, np.zeros(8, dtype=int)), 0.0)

    def test_roc_auc_matches_pairwise_with_ties(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = np.round(rng.random(200) + 0.3 * y, 1)  # many ties
        self.assertAlmostEqual(roc_auc(y, s), pairwise_auc(y, s), places=12)
        fpr, tpr, thr = roc_curve(y, s)
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0.0, 0.0, 1.0, 1.0))
        self.assertTrue(np.all(np.diff(thr) < 0))

    def test_precision_recall(self):
        y, s = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
        prec, rec, thr = precision_recall_curve(y, s)
        np.testing.assert_allclose(prec, [1.0, 0.5, 2 / 3, 0.5])
        np.testing.assert_allclose(rec, [0.5, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(average_precision(y, s), 5 / 6)

    def test_probability_metrics(self):
        y = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(log_loss(y, np.full(4, 0.5)), math.log(2))
        self.assertAlmostEqual(brier_score(y, np.full(4, 0.5)), 0.25)
        P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        self.assertAlmostEqual(log_loss(np.array([0, 1]), P),
                               -(math.log(0.7) + math.log(0.8)) / 2)
        p = np.array([0.05, 0.15, 0.95, 0.85, 0.5])
        mean_p, frac, counts = calibration_curve(np.array([0, 0, 1, 1, 1]), p, 10)
        self.assertEqual(counts.sum(), 5)
        np.testing.assert_allclose(mean_p, [0.05, 0.15, 0.5, 0.85, 0.95])
        np.testing.assert_allclose(frac, [0, 0, 1, 1, 1])


class TestSplitters(unittest.TestCase):
    def check_partition(self, splits, n):
        tests = [test for _, test in splits]
        np.testing.assert_array_equal(np.sort(np.concatenate(tests)), np.arange(n))
        for train, test in splits:
            self.assertEqual(len(np.intersect1d(train, test)), 0)
            self.assertEqual(len(train) + len(test), n)

    def test_kfold_on_table(self):
        table = {"a": np.arange(23.0), "b": np.arange(23.0)}
        splits = list(KFold(5, seed=1).split(table))
        self.assertEqual(len(splits), 5)
        self.check_partition(splits, 23)

    def test_stratified(self):
        y = np.array([0] * 80 + [1] * 20)
        splits = list(StratifiedKFold(5).split(np.zeros((100, 1)), y))
        self.check_partition(splits, 100)
        for _, test in splits:
            self.assertEqual(y[test].sum(), 4)

    def test_groups_and_time(self):
        groups = np.repeat(np.arange(10), 3)
        for train, test in GroupKFold(5).split(None, None, groups):
            self.assertEqual(len(np.intersect1d(groups[train], groups[test])), 0)
        for train, test in TimeSeriesSplit(4).split(np.zeros(50)):
            self.assertLess(train.max(), test.min())


class Probe(Estimator):
    """Records what it was fitted on, to test cross_validate."""

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.n_fit_ = len(y)
        return self

    def predict(self, X):
        return np.zeros(len(X["x"]), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X["x"]), 1))


class TestCrossValidation(unittest.TestCase):
    def test_dispatch_cloning_and_tables(self):
        table = {"x": np.arange(40.0)}
        y = np.array([0, 1] * 20)
        probe = Probe()
        seen = {}

        def gets_pred(y_true, out):
            seen["pred"] = out
            return 0.0

        def brier_like(y_true, out):
            seen["proba"] = out
            return 1.0

        brier_like.needs_proba = True
        scores = cross_validate(probe, table, y, cv=4,
                                scoring={"p": gets_pred, "q": brier_like,
                                         "roc_auc": roc_auc})
        self.assertFalse(hasattr(probe, "n_fit_"))  # the original is untouched
        self.assertEqual(len(scores["p"]), 4)
        np.testing.assert_array_equal(seen["pred"], 0)
        np.testing.assert_allclose(seen["proba"], 0.75)
        np.testing.assert_allclose(scores["roc_auc"], 0.5)

    def test_cross_val_predict_and_grid_search(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = X @ np.array([1.0, 2.0, 0.0]) + rng.normal(0, 0.1, 120)
        oof = cross_val_predict(Ridge(0.1), X, y, cv=5)
        self.assertEqual(oof.shape, (120,))
        self.assertLess(mse(y, oof), 0.05)
        search = grid_search(Ridge(), {"alpha": [1e-3, 100.0, 1e4]}, X, y,
                             cv=5, scoring=mse)
        self.assertEqual(search["best_params"], {"alpha": 1e-3})
        self.assertEqual(len(search["results"]), 3)
        self.assertTrue(hasattr(search["best_estimator"], "coef_"))

    def test_nested_params_and_churn(self):
        pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                         ("model", LogisticRegression())])
        set_param(pipe, "model__alpha", 0.5)
        self.assertEqual(dict(pipe.steps)["model"].alpha, 0.5)
        table, y = make_churn()
        X_tr, _, y_tr, _ = train_test_split(table, y, test_size=0.2, seed=0,
                                            stratify=True)
        scores = cross_validate(pipe, X_tr, y_tr, cv=StratifiedKFold(5),
                                scoring={"accuracy": accuracy, "roc_auc": roc_auc})
        self.assertTrue(0.6 < scores["accuracy"].mean() < 0.85)
        self.assertTrue(0.7 < scores["roc_auc"].mean() < 0.9)


class TestStatistics(unittest.TestCase):
    def test_bootstrap(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        pred = np.where(rng.random(500) < 0.8, y, 1 - y)
        est, low, high = bootstrap_ci(y, pred, accuracy, n_boot=500, seed=0)
        self.assertTrue(low <= est <= high)
        self.assertLess(high - low, 0.1)
        self.assertEqual(bootstrap_ci(y, pred, accuracy, n_boot=200, seed=3),
                         bootstrap_ci(y, pred, accuracy, n_boot=200, seed=3))
        diff, low, high = paired_bootstrap(y, pred, pred, accuracy, n_boot=100)
        self.assertEqual((diff, low, high), (0.0, 0.0, 0.0))

    def test_mcnemar(self):
        y = np.ones(20, dtype=int)
        a = np.r_[np.ones(10), np.ones(10)]
        b = np.r_[np.zeros(10), np.ones(10)]
        self.assertEqual(mcnemar(y, a, b), (10, 0, 2 / 1024))
        self.assertEqual(mcnemar(y, a, a)[2], 1.0)

    def test_t_test(self):
        self.assertAlmostEqual(_t_two_sided_p(12.706, 1), 0.05, places=4)
        self.assertAlmostEqual(_t_two_sided_p(2.262, 9), 0.05, places=3)
        a, b = np.array([0.80, 0.82, 0.81, 0.83]), np.array([0.79, 0.80, 0.80, 0.80])
        t0, p0 = corrected_resampled_ttest(a, b, 0.0)
        t1, p1 = corrected_resampled_ttest(a, b, 0.25)
        d = a - b
        self.assertAlmostEqual(t0, d.mean() / (d.std(ddof=1) / 2))  # ordinary paired t
        self.assertLess(abs(t1), abs(t0))
        self.assertGreater(p1, p0)


if __name__ == "__main__":
    unittest.main()
