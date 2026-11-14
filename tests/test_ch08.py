"""Tests for Chapter 8: decision trees and ensembles.

Run from Instructor Materials/Labs with:  python -m unittest tests.test_ch08
"""
import unittest

import numpy as np

from tinyml.base import Pipeline, clone
from tinyml.data import make_churn
from tinyml.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                             RandomForestClassifier)
from tinyml.preprocessing import TableEncoder, train_test_split
from tinyml.tree import (DecisionTreeClassifier, DecisionTreeRegressor, best_split,
                         entropy_cost, export_text, gini_cost, squared_error_cost)


class TestImpurity(unittest.TestCase):
    def test_known_values(self):
        counts = np.array([[5.0, 5.0], [10.0, 0.0], [2.0, 6.0]])
        np.testing.assert_allclose(gini_cost(counts), [5.0, 0.0, 8 * (1 - 0.25 ** 2 - 0.75 ** 2)])
        np.testing.assert_allclose(entropy_cost(counts)[:2], [10.0, 0.0])
        stats = np.array([[3.0, 6.0, 14.0]])  # y = 1, 2, 3: mean 2, SSE 2
        np.testing.assert_allclose(squared_error_cost(stats), [2.0])

    def test_best_split_on_separable_feature(self):
        x = np.array([3.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        y = np.array([0, 0, 0, 1, 1, 1])
        stats = np.eye(2)[y]
        cost, threshold = best_split(x, stats, gini_cost, min_leaf=1)
        self.assertAlmostEqual(cost, 0.0)
        self.assertAlmostEqual(threshold, 6.5)

    def test_best_split_respects_min_leaf_and_ties(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        stats = np.eye(2)[[1, 0, 0, 0]]
        _, threshold = best_split(x, stats, gini_cost, min_leaf=2)
        self.assertAlmostEqual(threshold, 2.5)
        self.assertIsNone(best_split(np.ones(4), stats, gini_cost, 1))


class TestTrees(unittest.TestCase):
    def test_xor_needs_depth_two(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        self.assertLess(np.mean(DecisionTreeClassifier(max_depth=1).fit(X, y).predict(X) == y), 1)
        np.testing.assert_array_equal(DecisionTreeClassifier(max_depth=2).fit(X, y).predict(X), y)

    def test_regressor_recovers_step_function(self):
        x = np.linspace(0, 1, 200)[:, None]
        y = np.where(x[:, 0] < 0.3, 1.0, np.where(x[:, 0] < 0.7, 5.0, -2.0))
        tree = DecisionTreeRegressor(max_depth=2).fit(x, y)
        np.testing.assert_allclose(tree.predict(x), y)
        self.assertEqual(int(np.sum(tree.left_ == -1)), 3)

    def test_attributes_and_importances(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        y = (X[:, 1] > 0.2).astype(int)
        tree = DecisionTreeClassifier(criterion="entropy").fit(X, y)
        self.assertEqual(tree.feature_[0], 1)
        self.assertAlmostEqual(tree.feature_importances_.sum(), 1.0)
        self.assertGreater(tree.feature_importances_[1], 0.99)
        proba = tree.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        leaves = tree.apply(X)
        self.assertTrue(np.all(tree.left_[leaves] == -1))
        text = export_text(tree, ["a", "b", "c"])
        self.assertIn("b <= ", text)
        self.assertIn("leaf: n=", text)

    def test_min_samples_leaf_and_depth(self):
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(200, 4)), rng.integers(0, 2, 200)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15).fit(X, y)
        leaves = tree.left_ == -1
        self.assertTrue(np.all(tree.n_node_samples_[leaves] >= 15))
        self.assertLessEqual(leaves.sum(), 16)

    def test_sample_weight_equals_duplication(self):
        rng = np.random.default_rng(2)
        X, y = rng.normal(size=(60, 2)), rng.integers(0, 2, 60)
        w = rng.integers(1, 4, 60)
        weighted = DecisionTreeClassifier(max_depth=3).fit(X, y, sample_weight=w)
        repeated = DecisionTreeClassifier(max_depth=3).fit(np.repeat(X, w, axis=0),
                                                           np.repeat(y, w))
        np.testing.assert_allclose(weighted.predict_proba(X), repeated.predict_proba(X))

    def test_max_features_and_seed(self):
        rng = np.random.default_rng(3)
        X, y = rng.normal(size=(100, 6)), rng.integers(0, 2, 100)
        a = DecisionTreeClassifier(max_features=2, seed=5).fit(X, y)
        b = clone(a).fit(X, y)
        np.testing.assert_array_equal(a.feature_, b.feature_)


class TestEnsembles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        table, y = make_churn(n=1500, seed=0)
        train, test, cls.y_train, cls.y_test = train_test_split(
            table, y, test_size=0.3, seed=0, stratify=True)
        enc = TableEncoder(["tenure_months", "monthly_charges", "support_calls"],
                           ["contract"]).fit(train)
        cls.X_train, cls.X_test = enc.transform(train), enc.transform(test)

    def test_random_forest(self):
        forest = RandomForestClassifier(n_estimators=20, min_samples_leaf=5).fit(
            self.X_train, self.y_train)
        self.assertTrue(0.6 < forest.oob_score_ < 0.9)
        proba = forest.predict_proba(self.X_test)
        self.assertEqual(proba.shape, (len(self.X_test), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        again = clone(forest).fit(self.X_train, self.y_train)
        np.testing.assert_allclose(again.predict_proba(self.X_test), proba)
        self.assertGreater(np.mean(forest.predict(self.X_test) == self.y_test), 0.65)

    def test_gradient_boosting(self):
        gb = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1,
                                        max_depth=2, subsample=0.8).fit(self.X_train, self.y_train)
        p = self.y_train.mean()
        self.assertAlmostEqual(gb.init_, np.log(p / (1 - p)))
        self.assertLess(gb.train_loss_[-1], gb.train_loss_[0])
        stages = list(gb.staged_decision_function(self.X_test))
        np.testing.assert_allclose(stages[-1], gb.decision_function(self.X_test))
        self.assertGreater(np.mean(gb.predict(self.X_test) == self.y_test), 0.65)

    def test_boosting_fits_a_separable_problem(self):
        x = np.linspace(-1, 1, 100)[:, None]
        y = (np.abs(x[:, 0]) < 0.5).astype(int)  # needs two thresholds
        ada = AdaBoostClassifier(n_estimators=30).fit(x, y)
        self.assertEqual(np.mean(ada.predict(x) == y), 1.0)
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5,
                                        max_depth=2).fit(x, y)
        self.assertEqual(np.mean(gb.predict(x) == y), 1.0)
        np.testing.assert_allclose(ada.predict_proba(x).sum(axis=1), 1.0)

    def test_in_pipeline(self):
        table, y = make_churn(n=600, seed=4)
        pipe = Pipeline([("encode", TableEncoder(["tenure_months"], ["contract"])),
                         ("model", GradientBoostingClassifier(n_estimators=10))])
        self.assertEqual(pipe.fit(table, y).predict_proba(table).shape, (600, 2))


if __name__ == "__main__":
    unittest.main()
