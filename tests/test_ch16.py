"""Tests for Chapter 16: tinyml/fairness.py, tinyml/explain.py, tinyml/privacy.py.

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch16
"""
import math
import unittest

import numpy as np

from tinyml.explain import neg_log_loss, partial_dependence, permutation_importance
from tinyml.fairness import (demographic_parity_difference, disparate_impact_ratio,
                             equalized_odds_difference, group_calibration, group_rates,
                             selection_rate, threshold_sweep, thresholds_for_tpr)
from tinyml.privacy import dp_count, dp_mean, laplace_mechanism


class TestFairness(unittest.TestCase):
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0, 1, 1, 1, 1])
    group = np.array(["a"] * 4 + ["b"] * 4)

    def test_group_rates_known_answer(self):
        r = group_rates(self.y, self.pred, self.group)
        self.assertEqual(r["a"]["n"], 4)
        self.assertAlmostEqual(r["a"]["selection_rate"], 0.5)
        self.assertAlmostEqual(r["a"]["tpr"], 0.5)
        self.assertAlmostEqual(r["a"]["fpr"], 0.5)
        self.assertAlmostEqual(r["a"]["precision"], 0.5)
        self.assertAlmostEqual(r["b"]["selection_rate"], 1.0)
        self.assertAlmostEqual(r["b"]["tpr"], 1.0)
        self.assertAlmostEqual(r["b"]["fpr"], 1.0)

    def test_summary_metrics(self):
        self.assertEqual(selection_rate(self.pred, self.group), {"a": 0.5, "b": 1.0})
        self.assertAlmostEqual(demographic_parity_difference(self.pred, self.group), 0.5)
        self.assertAlmostEqual(disparate_impact_ratio(self.pred, self.group), 0.5)
        self.assertAlmostEqual(equalized_odds_difference(self.y, self.pred, self.group), 0.5)

    def test_threshold_sweep_shape_and_monotone(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        score = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
        group = rng.choice(["a", "b"], 500)
        sweep = threshold_sweep(y, score, group)
        self.assertEqual(sweep["a"].shape, (19, 4))
        self.assertTrue(np.all(np.diff(sweep["a"][:, 1]) <= 0))  # selection falls

    def test_thresholds_for_tpr(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 2000)
        group = rng.choice(["a", "b"], 2000)
        score = 0.4 * y + rng.random(2000) * 0.6 + 0.1 * (group == "b")
        cut = thresholds_for_tpr(y, score, group, 0.8)
        for g in ("a", "b"):
            m = (group == g) & (y == 1)
            tpr = np.mean(score[m] >= cut[g])
            self.assertGreaterEqual(tpr, 0.8)
            self.assertLess(tpr, 0.8 + 2 / m.sum())

    def test_group_calibration_of_true_probabilities(self):
        rng = np.random.default_rng(2)
        score = rng.random(40000)
        y = (rng.random(40000) < score).astype(int)
        group = rng.choice(["a", "b"], 40000)
        for bins in group_calibration(y, score, group).values():
            for mean_score, rate, n in bins:
                self.assertLess(abs(mean_score - rate), 0.03)


class LinearInA:
    """A fake model whose P(y=1) depends only on column 'a'."""

    def predict_proba(self, table):
        p = 1 / (1 + np.exp(-2 * np.asarray(table["a"], dtype=float)))
        return np.column_stack([1 - p, p])


class TestExplain(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=3000)
        self.table = {"a": a, "b": rng.normal(size=3000),
                      "c": rng.choice(["x", "y"], 3000)}
        self.y = (rng.random(3000) < 1 / (1 + np.exp(-2 * a))).astype(int)

    def test_unused_columns_have_zero_importance(self):
        result = permutation_importance(LinearInA(), self.table, self.y, n_repeats=3)
        self.assertEqual(list(result)[0], "a")
        self.assertGreater(result["a"][0], 0.1)
        self.assertEqual(result["b"], (0.0, 0.0))
        self.assertEqual(result["c"], (0.0, 0.0))

    def test_neg_log_loss_known_answer(self):
        self.assertAlmostEqual(neg_log_loss(np.array([1, 0]), np.array([0.5, 0.5])),
                               math.log(0.5))

    def test_partial_dependence(self):
        grid, avg = partial_dependence(LinearInA(), self.table, "a", grid=[0.0, 1.0])
        np.testing.assert_allclose(avg, [0.5, 1 / (1 + math.exp(-2))])
        grid, avg = partial_dependence(LinearInA(), self.table, "c")
        self.assertEqual(list(grid), ["x", "y"])
        self.assertAlmostEqual(avg[0], avg[1])
        grid, _ = partial_dependence(LinearInA(), self.table, "b")
        self.assertEqual(len(grid), 10)


class TestPrivacy(unittest.TestCase):
    def test_laplace_noise_has_the_right_scale(self):
        rng = np.random.default_rng(0)
        draws = np.array([laplace_mechanism(0.0, 2.0, 0.5, rng) for _ in range(40000)])
        self.assertAlmostEqual(draws.mean(), 0.0, delta=0.1)
        self.assertAlmostEqual(np.mean(np.abs(draws)), 4.0, delta=0.1)  # E|X| = b

    def test_invalid_arguments(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            laplace_mechanism(1.0, 1.0, 0.0, rng)
        with self.assertRaises(ValueError):
            laplace_mechanism(1.0, -1.0, 1.0, rng)

    def test_dp_mean_clips_and_converges(self):
        rng = np.random.default_rng(0)
        x = np.array([0.0, 10.0, 200.0, 50.0])
        self.assertAlmostEqual(dp_mean(x, 0, 100, 1e9, rng), 40.0, places=4)

    def test_dp_mean_error_matches_theory(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 1000)
        errors = np.array([dp_mean(x, 0, 1, 0.1, rng) - x.mean() for _ in range(5000)])
        expected_rmse = math.sqrt(2) * (1 / 1000) / 0.1
        self.assertAlmostEqual(np.sqrt(np.mean(errors ** 2)), expected_rmse,
                               delta=0.05 * expected_rmse)

    def test_dp_count(self):
        rng = np.random.default_rng(2)
        counts = [dp_count(np.ones(50, bool), 1.0, rng) for _ in range(4000)]
        self.assertAlmostEqual(np.mean(counts), 50.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
