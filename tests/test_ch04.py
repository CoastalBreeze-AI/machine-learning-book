"""Tests for Chapter 4: optimizers, the gradient checker, and LinearRegression.

Run from Instructor Materials/Labs with:  python -m unittest tests.test_ch04
"""
import unittest

import numpy as np

from tinyml.base import Pipeline, clone
from tinyml.linear import LinearRegression, add_intercept, mse_objective
from tinyml.optimize import (check_gradient, gradient_descent, minibatch_sgd,
                             numerical_gradient)
from tinyml.preprocessing import StandardScaler


def make_regression(n=300, d=3, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.arange(1.0, d + 1.0)
    return X, X @ w + 2.0 + noise * rng.normal(size=n), w


def quadratic(w):
    """f(w) = 0.5 * (w0^2 + 10 * w1^2), minimized at the origin."""
    scales = np.array([1.0, 10.0])
    return 0.5 * np.sum(scales * w ** 2), scales * w


class TestOptimize(unittest.TestCase):
    def test_gradient_descent_converges_on_quadratic(self):
        w, history = gradient_descent(quadratic, [3.0, -2.0], lr=0.1, n_iter=300)
        np.testing.assert_allclose(w, [0.0, 0.0], atol=1e-8)
        self.assertEqual(len(history), 301)
        self.assertTrue(np.all(np.diff(history) <= 1e-12))  # monotone for lr < 2/L

    def test_gradient_descent_diverges_above_two_over_L(self):
        # L = 10, so any lr > 0.2 diverges along w1.
        _, history = gradient_descent(quadratic, [1.0, 1.0], lr=0.21, n_iter=200)
        self.assertGreater(history[-1], history[0])

    def test_tol_stops_early(self):
        _, history = gradient_descent(quadratic, [1.0, 1.0], lr=0.1, n_iter=10_000,
                                      tol=1e-6)
        self.assertLess(len(history), 10_001)

    def test_momentum_is_faster_on_ill_conditioned_quadratic(self):
        _, plain = gradient_descent(quadratic, [1.0, 1.0], lr=0.1, n_iter=60)
        _, heavy = gradient_descent(quadratic, [1.0, 1.0], lr=0.1, n_iter=60,
                                    momentum=0.5)
        self.assertLess(heavy[-1], plain[-1])

    def test_minibatch_sgd(self):
        X, y, _ = make_regression()
        A = add_intercept(X)

        def objective(w, A_batch, y_batch):
            return mse_objective(A_batch, y_batch)(w)

        w, history = minibatch_sgd(objective, np.zeros(4), A, y, lr=0.05,
                                   n_epochs=30, batch_size=16, seed=0)
        self.assertEqual(len(history), 31)
        np.testing.assert_allclose(w, [2.0, 1.0, 2.0, 3.0], atol=0.05)
        # Full batch with one step per epoch is gradient descent.
        w_full, _ = minibatch_sgd(objective, np.zeros(4), A, y, lr=0.1,
                                  n_epochs=5, batch_size=len(y))
        w_gd, _ = gradient_descent(mse_objective(A, y), np.zeros(4), lr=0.1, n_iter=5)
        np.testing.assert_allclose(w_full, w_gd)

    def test_numerical_gradient_and_checker(self):
        rng = np.random.default_rng(1)
        w = rng.normal(size=2)
        np.testing.assert_allclose(numerical_gradient(lambda v: quadratic(v)[0], w),
                                   quadratic(w)[1], rtol=1e-6)
        self.assertLess(check_gradient(quadratic, w), 1e-7)
        bad = lambda v: (quadratic(v)[0], 2 * quadratic(v)[1])
        self.assertGreater(check_gradient(bad, w), 1e-2)
        self.assertEqual(check_gradient(quadratic, np.zeros(2)), 0.0)

    def test_mse_objective_gradient(self):
        X, y, _ = make_regression(n=50)
        objective = mse_objective(add_intercept(X), y)
        w = np.random.default_rng(2).normal(size=4)
        self.assertLess(check_gradient(objective, w), 1e-6)


class TestLinearRegression(unittest.TestCase):
    def test_normal_equations_recover_weights(self):
        X, y, w = make_regression(noise=0.0)
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_, w, atol=1e-10)
        self.assertAlmostEqual(model.intercept_, 2.0, places=10)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-9)

    def test_gd_matches_normal_equations(self):
        X, y, _ = make_regression()
        exact = LinearRegression().fit(X, y)
        gd = LinearRegression(method="gd", lr=0.2, n_iter=2000).fit(X, y)
        np.testing.assert_allclose(gd.coef_, exact.coef_, atol=1e-6)
        self.assertAlmostEqual(gd.intercept_, exact.intercept_, places=6)
        self.assertEqual(len(gd.loss_history_), 2001)

    def test_one_dimensional_input_and_collinear_columns(self):
        x = np.linspace(0, 1, 20)
        model = LinearRegression().fit(x, 3 * x + 1)
        self.assertAlmostEqual(model.coef_[0], 3.0)
        self.assertEqual(model.predict(x).shape, (20,))
        X = np.column_stack([x, 2 * x])  # exactly collinear: lstsq still works
        model = LinearRegression().fit(X, 3 * x + 1)
        np.testing.assert_allclose(model.predict(X), 3 * x + 1, atol=1e-9)

    def test_interface(self):
        model = LinearRegression(method="gd", lr=0.05, n_iter=10)
        self.assertEqual(model.get_params(), {"method": "gd", "lr": 0.05, "n_iter": 10})
        self.assertEqual(clone(model).lr, 0.05)
        with self.assertRaises(ValueError):
            LinearRegression(method="magic").fit(np.zeros((3, 1)), np.zeros(3))
        X, y, _ = make_regression()
        X_raw = X * [100, 1, 1] + [0, 0, 50]  # badly scaled columns
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", LinearRegression("gd", lr=0.3, n_iter=500))])
        exact = LinearRegression().fit(X, y)
        np.testing.assert_allclose(pipe.fit(X_raw, y).predict(X_raw),
                                   exact.predict(X), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
