"""Tests for Chapter 3: the estimator interface, the churn data, and preprocessing.

Run from Instructor Materials/Labs with:  python -m unittest tests.test_ch03
"""
import unittest

import numpy as np

from tinyml.base import Estimator, Pipeline, Transformer, clone
from tinyml.data import make_churn
from tinyml.preprocessing import (OneHotEncoder, SimpleImputer, StandardScaler,
                                  TableEncoder, train_test_split)

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]


class MeanModel(Estimator):
    """Predicts the training mean of y; used to test the interface."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y=None):
        self.mean_ = float(np.mean(y)) + self.shift
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class TinyLogistic(Estimator):
    """The stand-in logistic regression from Listing 3.15."""

    def __init__(self, lr=0.5, n_iter=500):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.w_ = np.zeros(A.shape[1])
        for _ in range(self.n_iter):
            p = 1.0 / (1.0 + np.exp(-A @ self.w_))
            self.w_ -= self.lr * A.T @ (p - y) / len(y)
        return self

    def predict(self, X):
        A = np.column_stack([np.ones(len(X)), X])
        return (A @ self.w_ > 0).astype(int)


class TestBase(unittest.TestCase):
    def test_params_and_clone(self):
        model = MeanModel(shift=2.0)
        self.assertEqual(model.get_params(), {"shift": 2.0})
        model.set_params(shift=3.0)
        self.assertEqual(model.shift, 3.0)
        with self.assertRaises(ValueError):
            model.set_params(nonsense=1)
        model.fit(np.zeros((3, 1)), np.array([1.0, 2.0, 3.0]))
        copy = clone(model)
        self.assertEqual(copy.shift, 3.0)
        self.assertFalse(hasattr(copy, "mean_"))
        self.assertEqual(repr(copy), "MeanModel(shift=3.0)")

    def test_fit_returns_self(self):
        model = MeanModel()
        self.assertIs(model.fit(np.zeros((2, 1)), np.array([0.0, 1.0])), model)

    def test_pipeline_fits_on_training_data_only(self):
        table, y = make_churn(n=1000)
        train, test, y_train, _ = train_test_split(table, y, seed=0)
        pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                         ("model", MeanModel())])
        pipe.fit(train, y_train)
        np.testing.assert_allclose(pipe["encode"].scaler_.mean_[0],
                                   train["tenure_months"].mean())
        self.assertAlmostEqual(pipe["model"].mean_, y_train.mean())
        # The unfitted template steps are untouched; fitted copies are used.
        self.assertFalse(hasattr(pipe.steps[0][1], "scaler_"))
        self.assertEqual(pipe.predict(test).shape, (200,))

    def test_transformer_fit_transform(self):
        X = np.array([[1.0], [3.0]])
        np.testing.assert_allclose(StandardScaler().fit_transform(X), [[-1.0], [1.0]])
        self.assertTrue(issubclass(StandardScaler, Transformer))

    def test_estimator_without_init_has_no_params(self):
        # get_params skips object.__init__'s *args and **kwargs, so a class
        # with no constructor of its own can be cloned and used in a Pipeline.
        self.assertEqual(StandardScaler().get_params(), {})
        self.assertIsInstance(clone(StandardScaler()), StandardScaler)
        X = np.array([[1.0], [3.0]])
        pipe = Pipeline([("scale", StandardScaler()), ("scale2", StandardScaler())])
        np.testing.assert_allclose(pipe.fit(X).transform(X), [[-1.0], [1.0]])


class TestData(unittest.TestCase):
    def test_shape_and_rates(self):
        table, y = make_churn()
        self.assertEqual(len(y), 5000)
        self.assertEqual(set(table), set(NUMERIC + CATEGORICAL + ["retention_offer_days"]))
        self.assertAlmostEqual(y.mean(), 0.321, places=3)
        self.assertTrue(all(len(col) == 5000 for col in table.values()))
        _, y_drift = make_churn(drift=True)
        self.assertAlmostEqual(y_drift.mean(), 0.446, delta=0.002)

    def test_reproducible(self):
        a, ya = make_churn(n=100, seed=3)
        b, yb = make_churn(n=100, seed=3)
        np.testing.assert_array_equal(ya, yb)
        np.testing.assert_array_equal(a["contract"], b["contract"])

    def test_total_charges_missing_at_random_given_tenure(self):
        table, _ = make_churn()
        missing = np.isnan(table["total_charges"])
        short = table["tenure_months"] <= 2
        self.assertGreater(missing[short].mean(), 0.25)
        self.assertLess(missing[~short].mean(), 0.05)

    def test_leaky_column_tracks_label(self):
        table, y = make_churn()
        offer = ~np.isnan(table["retention_offer_days"])
        self.assertGreater(y[offer].mean(), 0.85)
        self.assertLess(y[~offer].mean(), 0.1)


class TestPreprocessing(unittest.TestCase):
    def test_split_sizes_and_stratification(self):
        table, y = make_churn()
        train, test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                        seed=0, stratify=True)
        self.assertEqual((len(y_train), len(y_test)), (4000, 1000))
        self.assertEqual(len(train["age"]), 4000)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), places=2)

    def test_split_arrays_disjoint(self):
        X = np.arange(50).reshape(25, 2)
        y = np.arange(25)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=1)
        self.assertEqual(len(y_te), 5)
        self.assertEqual(len(set(y_tr) & set(y_te)), 0)
        np.testing.assert_array_equal(X_te[:, 0] // 2, y_te)

    def test_imputer(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
        filled = SimpleImputer().fit(X).transform(X)
        np.testing.assert_allclose(filled, [[1.0, 6.0], [3.0, 4.0], [2.0, 8.0]])
        mean_filled = SimpleImputer("mean").fit(X).transform(X)
        self.assertAlmostEqual(mean_filled[0, 1], 6.0)

    def test_scaler_constant_column(self):
        X = np.array([[1.0, 5.0], [3.0, 5.0]])
        Z = StandardScaler().fit(X).transform(X)
        np.testing.assert_allclose(Z, [[-1.0, 0.0], [1.0, 0.0]])

    def test_onehot_unknown_category(self):
        encoder = OneHotEncoder().fit(np.array(["a", "b", "a"]))
        np.testing.assert_array_equal(encoder.transform(np.array(["b", "z"])),
                                      [[0.0, 1.0], [0.0, 0.0]])

    def test_table_encoder(self):
        table, y = make_churn(n=500)
        encoder = TableEncoder(NUMERIC, CATEGORICAL, missing_indicators=True).fit(table)
        M = encoder.transform(table)
        self.assertEqual(M.shape, (500, len(encoder.feature_names_)))
        self.assertIn("total_charges_missing", encoder.feature_names_)
        self.assertIn("contract=two-year", encoder.feature_names_)
        self.assertFalse(np.isnan(M).any())
        np.testing.assert_allclose(M[:, :5].mean(axis=0), 0.0, atol=1e-9)


class TestLeakage(unittest.TestCase):
    def test_leaky_column_inflates_accuracy(self):
        table, y = make_churn()
        train, test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                        seed=0, stratify=True)
        honest = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                           ("model", TinyLogistic())]).fit(train, y_train)
        leaky = Pipeline([("encode", TableEncoder(NUMERIC + ["retention_offer_days"],
                                                  CATEGORICAL, missing_indicators=True)),
                          ("model", TinyLogistic())]).fit(train, y_train)
        honest_acc = np.mean(honest.predict(test) == y_test)
        leaky_acc = np.mean(leaky.predict(test) == y_test)
        self.assertAlmostEqual(honest_acc, 0.746, places=3)
        self.assertAlmostEqual(leaky_acc, 0.921, places=3)
        deployed = dict(test, retention_offer_days=np.full(len(y_test), np.nan))
        self.assertLess(np.mean(leaky.predict(deployed) == y_test), honest_acc)


if __name__ == "__main__":
    unittest.main()
