"""Tests for Chapter 9: tinyml.naive_bayes and tinyml.bayes."""
import unittest

import numpy as np

from tinyml.base import clone
from tinyml.bayes import BayesianLinearRegression
from tinyml.data import make_churn
from tinyml.naive_bayes import CategoricalNB, GaussianNB, MixedNB, log_normalize_rows
from tinyml.preprocessing import train_test_split


def normal_logpdf(x, mean, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


class TestGaussianNB(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = np.vstack([rng.normal([0, 0], 1, (40, 2)), rng.normal([3, 1], 1, (60, 2))])
        self.y = np.repeat([0, 1], [40, 60])

    def test_parameters(self):
        model = GaussianNB(var_smoothing=0.0).fit(self.X, self.y)
        np.testing.assert_allclose(model.theta_[1], self.X[40:].mean(axis=0))
        np.testing.assert_allclose(model.var_[0], self.X[:40].var(axis=0))
        np.testing.assert_allclose(np.exp(model.class_log_prior_), [0.4, 0.6])

    def test_known_answer_posterior(self):
        model = GaussianNB(var_smoothing=0.0).fit(self.X, self.y)
        x = np.array([[1.0, 0.5]])
        joint = [np.log(p) + sum(normal_logpdf(x[0, j], self.X[rows, j].mean(), self.X[rows, j].var())
                                 for j in range(2))
                 for p, rows in [(0.4, slice(0, 40)), (0.6, slice(40, 100))]]
        expected = np.exp(joint - np.logaddexp(*joint))
        np.testing.assert_allclose(model.predict_proba(x)[0], expected)

    def test_probabilities_and_predictions(self):
        model = GaussianNB().fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(model.predict(self.X), model.classes_[proba.argmax(axis=1)])
        self.assertGreater(np.mean(model.predict(self.X) == self.y), 0.85)

    def test_missing_feature_drops_out(self):
        full = GaussianNB(var_smoothing=0.0).fit(self.X, self.y)
        first_only = GaussianNB(var_smoothing=0.0).fit(self.X[:, :1], self.y)
        X_missing = self.X.copy()
        X_missing[:, 1] = np.nan
        np.testing.assert_allclose(full.predict_proba(X_missing), first_only.predict_proba(self.X[:, :1]))

    def test_extreme_scores_do_not_overflow(self):
        rows = log_normalize_rows(np.array([[-1000.0, -1001.0], [800.0, 0.0]]))
        np.testing.assert_allclose(np.exp(rows).sum(axis=1), 1.0)
        self.assertTrue(np.all(np.isfinite(rows)))

    def test_clone(self):
        model = clone(GaussianNB(var_smoothing=1e-3))
        self.assertEqual(model.get_params(), {"var_smoothing": 1e-3})


class TestCategoricalNB(unittest.TestCase):
    def test_smoothed_counts(self):
        X = np.array([["a"], ["a"], ["b"], ["b"], ["b"], ["c"]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = CategoricalNB(alpha=1.0).fit(X, y)
        # Class 0 saw a, a, b: (2+1)/(3+3), (1+1)/6, (0+1)/6.
        np.testing.assert_allclose(np.exp(model.feature_log_prob_[0][0]), [3 / 6, 2 / 6, 1 / 6])
        proba = model.predict_proba(np.array([["a"]]))[0]
        expected = np.array([0.5 * 3 / 6, 0.5 * 1 / 6])
        np.testing.assert_allclose(proba, expected / expected.sum())

    def test_unseen_category_gives_prior(self):
        X = np.array([["a"], ["b"], ["b"], ["b"]])
        y = np.array([0, 0, 1, 1])
        model = CategoricalNB().fit(X, y)
        np.testing.assert_allclose(model.predict_proba(np.array([["zzz"]]))[0], [0.5, 0.5])


class TestMixedNB(unittest.TestCase):
    def test_on_churn(self):
        table, y = make_churn(n=2000, seed=1)
        X_train, X_test, y_train, y_test = train_test_split(table, y, seed=0, stratify=True)
        numeric = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
        categorical = ["contract", "payment_method", "region"]
        model = MixedNB(numeric, categorical).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        self.assertEqual(proba.shape, (len(y_test), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        self.assertGreater(np.mean(model.predict(X_test) == y_test), 0.6)
        # The joint log-likelihood is the sum of the two parts plus one prior.
        num = np.column_stack([X_test[c] for c in numeric])
        cat = np.column_stack([X_test[c] for c in categorical])
        manual = (model.class_log_prior_ + model.gaussian_._feature_log_likelihood(num)
                  + model.categorical_._feature_log_likelihood(cat))
        np.testing.assert_allclose(model.predict_log_proba(X_test), log_normalize_rows(manual))


class TestBayesianLinearRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.uniform(-1, 1, (30, 2))
        self.y = 1.0 + self.X @ np.array([2.0, -1.0]) + rng.normal(0, 0.2, 30)

    def test_posterior_mean_is_ridge(self):
        alpha, beta = 2.0, 25.0
        model = BayesianLinearRegression(alpha, beta).fit(self.X, self.y)
        Phi = np.column_stack([np.ones(30), self.X])
        ridge = np.linalg.solve(Phi.T @ Phi + alpha / beta * np.eye(3), Phi.T @ self.y)
        np.testing.assert_allclose(model.mean_, ridge)
        self.assertAlmostEqual(model.intercept_, ridge[0])
        np.testing.assert_allclose(model.coef_, ridge[1:])

    def test_predictive_std(self):
        model = BayesianLinearRegression(1.0, 25.0).fit(self.X, self.y)
        mean, std = model.predict(np.array([[0.0, 0.0], [5.0, 5.0], [50.0, 50.0]]), return_std=True)
        self.assertEqual(mean.shape, (3,))
        self.assertTrue(np.all(std > 0.2))            # never below the noise level
        self.assertLess(std[0], std[1])
        self.assertLess(std[1], std[2])               # grows away from the data
        np.testing.assert_allclose(model.predict(np.zeros((1, 2))), mean[:1])

    def test_log_evidence_matches_marginal_gaussian(self):
        alpha, beta = 0.5, 10.0
        model = BayesianLinearRegression(alpha, beta).fit(self.X, self.y)
        Phi = np.column_stack([np.ones(30), self.X])
        cov = np.eye(30) / beta + Phi @ Phi.T / alpha
        _, logdet = np.linalg.slogdet(cov)
        direct = -0.5 * (self.y @ np.linalg.solve(cov, self.y) + logdet + 30 * np.log(2 * np.pi))
        self.assertAlmostEqual(model.log_evidence(self.X, self.y), direct, places=8)

    def test_samples_and_no_intercept(self):
        model = BayesianLinearRegression(1.0, 25.0).fit(self.X, self.y)
        draws = model.sample_weights(20000, seed=0)
        np.testing.assert_allclose(draws.mean(axis=0), model.mean_, atol=0.01)
        plain = BayesianLinearRegression(1.0, 25.0, fit_intercept=False).fit(self.X, self.y)
        self.assertEqual(plain.mean_.shape, (2,))
        self.assertEqual(plain.intercept_, 0.0)


if __name__ == "__main__":
    unittest.main()
