"""Tests for the Chapter 5 models in tinyml/linear.py.

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch05
"""
import unittest

import numpy as np

from tinyml.base import Pipeline, clone
from tinyml.data import make_churn
from tinyml.linear import (Lasso, LogisticRegression, PolynomialFeatures, Ridge,
                           SoftmaxRegression, sigmoid, soft_threshold, softmax)
from tinyml.preprocessing import StandardScaler, TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]


def regression_data(n=200, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([2.0, -1.0, 0.0, 0.5, 0.0])[:d]
    return X, X @ w + 3.0 + rng.normal(0, 0.1, n), w


class TestPolynomialFeatures(unittest.TestCase):
    def test_columns_and_values(self):
        X = np.array([[2.0, 3.0], [1.0, -1.0]])
        poly = PolynomialFeatures(2).fit(X)
        expected = np.array([[2, 3, 4, 6, 9], [1, -1, 1, -1, 1]], dtype=float)
        np.testing.assert_allclose(poly.transform(X), expected)
        self.assertEqual(poly.feature_names(["a", "b"]),
                         ["a", "b", "a*a", "a*b", "b*b"])

    def test_one_dimensional_input(self):
        out = PolynomialFeatures(3).fit_transform(np.array([1.0, 2.0]))
        np.testing.assert_allclose(out, [[1, 1, 1], [2, 4, 8]])


class TestRidge(unittest.TestCase):
    def test_alpha_zero_is_least_squares(self):
        X, y, _ = regression_data()
        A = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        model = Ridge(0.0).fit(X, y)
        np.testing.assert_allclose(model.coef_, w[1:], atol=1e-8)
        self.assertAlmostEqual(model.intercept_, w[0], places=8)

    def test_closed_form_and_unpenalized_intercept(self):
        X, y, _ = regression_data()
        alpha = 5.0
        Xc, yc = X - X.mean(0), y - y.mean()
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
        model = Ridge(alpha).fit(X, y)
        np.testing.assert_allclose(model.coef_, w, atol=1e-10)
        shifted = Ridge(alpha).fit(X, y + 100.0)  # only the intercept moves
        np.testing.assert_allclose(shifted.coef_, model.coef_, atol=1e-10)
        self.assertAlmostEqual(shifted.intercept_ - model.intercept_, 100.0)

    def test_shrinkage_is_monotone(self):
        X, y, _ = regression_data()
        norms = [np.linalg.norm(Ridge(a).fit(X, y).coef_) for a in [0, 1, 10, 100, 1000]]
        self.assertTrue(all(a > b for a, b in zip(norms, norms[1:])))


class TestLasso(unittest.TestCase):
    def test_soft_threshold(self):
        np.testing.assert_allclose(soft_threshold(np.array([-3.0, -0.5, 0.2, 2.0]), 1.0),
                                   [-2.0, 0.0, 0.0, 1.0])

    def test_orthonormal_design_known_answer(self):
        # With centered, orthogonal columns scaled so Xc'Xc / n = I, the lasso
        # solution is the soft-thresholded least-squares solution.
        rng = np.random.default_rng(1)
        n = 100
        Z = rng.normal(size=(n, 3))
        Q, _ = np.linalg.qr(Z - Z.mean(axis=0))  # centered, orthonormal columns
        X = Q * np.sqrt(n)
        y = X @ np.array([1.0, -0.3, 0.05]) + 2.0
        ols = X.T @ (y - y.mean()) / n
        model = Lasso(alpha=0.1, n_iter=5000, tol=1e-12).fit(X, y)
        np.testing.assert_allclose(model.coef_, soft_threshold(ols, 0.1), atol=1e-8)
        self.assertEqual(model.coef_[2], 0.0)

    def test_kkt_conditions(self):
        X, y, _ = regression_data()
        alpha = 0.05
        model = Lasso(alpha, n_iter=5000, tol=1e-10).fit(X, y)
        r = y - model.predict(X)
        corr = (X - X.mean(0)).T @ r / len(y)
        active = model.coef_ != 0
        np.testing.assert_allclose(corr[active], alpha * np.sign(model.coef_[active]),
                                   atol=1e-6)
        self.assertTrue(np.all(np.abs(corr[~active]) <= alpha + 1e-6))
        self.assertTrue(np.all(Lasso(100.0).fit(X, y).coef_ == 0))


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.X = rng.normal(size=(500, 3))
        z = self.X @ np.array([1.5, -2.0, 0.0]) + 0.5
        self.y = (rng.random(500) < sigmoid(z)).astype(int)

    def test_sigmoid_is_stable(self):
        np.testing.assert_allclose(sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0, 0.5, 1])

    def test_gradient_is_zero_at_solution(self):
        alpha = 0.01
        m = LogisticRegression(alpha=alpha).fit(self.X, self.y)
        p = m.predict_proba(self.X)[:, 1]
        grad_w = self.X.T @ (p - self.y) / len(self.y) + alpha * m.coef_
        self.assertLess(np.abs(grad_w).max(), 1e-8)
        self.assertLess(abs(np.mean(p - self.y)), 1e-8)  # intercept unpenalized

    def test_newton_and_gradient_descent_agree(self):
        newton = LogisticRegression(alpha=0.01).fit(self.X, self.y)
        gd = LogisticRegression(alpha=0.01, method="gd", lr=1.0, n_iter=5000).fit(self.X, self.y)
        np.testing.assert_allclose(gd.coef_, newton.coef_, atol=1e-4)

    def test_labels_probabilities_and_C(self):
        labels = np.where(self.y == 1, "yes", "no")
        m = LogisticRegression().fit(self.X, labels)
        self.assertEqual(list(m.classes_), ["no", "yes"])
        proba = m.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        self.assertTrue(set(m.predict(self.X)) <= {"no", "yes"})
        a = LogisticRegression(C=2.0).fit(self.X, self.y)
        b = LogisticRegression(alpha=1 / (2.0 * len(self.y))).fit(self.X, self.y)
        np.testing.assert_allclose(a.coef_, b.coef_)
        with self.assertRaises(ValueError):
            LogisticRegression().fit(self.X, np.arange(500) % 3)

    def test_churn_pipeline(self):
        table, y = make_churn()
        X_tr, X_te, y_tr, y_te = train_test_split(table, y, test_size=0.2, seed=0,
                                                  stratify=True)
        pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                         ("model", LogisticRegression())]).fit(X_tr, y_tr)
        self.assertAlmostEqual(np.mean(pipe.predict(X_te) == y_te), 0.744, delta=0.01)
        self.assertEqual(pipe.predict_proba(X_te).shape, (1000, 2))


class TestSoftmaxRegression(unittest.TestCase):
    def test_softmax_rows(self):
        P = softmax(np.array([[1000.0, 0.0, -1000.0], [1.0, 1.0, 1.0]]))
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        np.testing.assert_allclose(P[1], 1 / 3)

    def test_three_blobs(self):
        rng = np.random.default_rng(3)
        centers = np.array([[0, 3], [3, -2], [-3, -2]])
        y = np.repeat(np.array(["a", "b", "c"]), 100)
        X = centers[np.repeat(np.arange(3), 100)] + rng.normal(size=(300, 2))
        m = SoftmaxRegression().fit(X, y)
        self.assertGreater(np.mean(m.predict(X) == y), 0.95)
        self.assertEqual(m.coef_.shape, (2, 3))

    def test_two_classes_match_logistic(self):
        rng = np.random.default_rng(4)
        X = StandardScaler().fit_transform(rng.normal(size=(400, 3)))
        y = (rng.random(400) < sigmoid(X @ np.array([1.0, -1.0, 0.5]))).astype(int)
        soft = SoftmaxRegression(alpha=0.02, lr=1.0, n_iter=50000, tol=1e-10).fit(X, y)
        logit = LogisticRegression(alpha=0.01).fit(X, y)
        np.testing.assert_allclose(soft.coef_[:, 1] - soft.coef_[:, 0], logit.coef_,
                                   atol=1e-5)


class TestInterface(unittest.TestCase):
    def test_clone_and_params(self):
        for model in [PolynomialFeatures(3), Ridge(2.0), Lasso(0.1, n_iter=10),
                      LogisticRegression(alpha=0.1), SoftmaxRegression(lr=0.1)]:
            copy = clone(model)
            self.assertEqual(copy.get_params(), model.get_params())


if __name__ == "__main__":
    unittest.main()
