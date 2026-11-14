"""Tests for Chapter 7: nearest neighbors and kernel SVMs.

Run from Instructor Materials/Labs with:  python -m unittest tests.test_ch07
"""
import unittest

import numpy as np

from tinyml.base import Pipeline, clone
from tinyml.neighbors import (KNNClassifier, KNNRegressor, nearest_neighbors,
                              pairwise_distances)
from tinyml.svm import (KernelSVM, linear_kernel, polynomial_kernel, rbf_kernel,
                        smo)


def circles(n, seed):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    label = np.arange(n) % 2
    radius = np.where(label == 1, 0.5, 1.0)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    return X + rng.normal(0, 0.1, X.shape), label


class TestDistances(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.A, self.B = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))

    def test_metrics_match_brute_force(self):
        for metric, f in [
            ("euclidean", lambda a, b: np.linalg.norm(a - b)),
            ("manhattan", lambda a, b: np.abs(a - b).sum()),
            ("cosine", lambda a, b: 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
        ]:
            brute = np.array([[f(a, b) for b in self.B] for a in self.A])
            np.testing.assert_allclose(pairwise_distances(self.A, self.B, metric),
                                       brute, atol=1e-10)

    def test_unknown_metric(self):
        with self.assertRaises(ValueError):
            pairwise_distances(self.A, self.B, "chebyshev")

    def test_nearest_neighbors_sorted_and_batched(self):
        rng = np.random.default_rng(1)
        X, Q = rng.normal(size=(50, 3)), rng.normal(size=(23, 3))
        dist, index = nearest_neighbors(X, Q, k=4, batch_size=5)
        D = pairwise_distances(Q, X)
        np.testing.assert_array_equal(index, np.argsort(D, axis=1)[:, :4])
        self.assertTrue(np.all(np.diff(dist, axis=1) >= 0))
        self.assertEqual(dist.shape, (23, 4))


class TestKNN(unittest.TestCase):
    def test_one_neighbor_memorizes_training_set(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(40, 2)), rng.integers(0, 3, 40)
        model = KNNClassifier(k=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_probabilities_and_string_labels(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0]])
        y = np.array(["a", "a", "b", "b"])
        model = KNNClassifier(k=3).fit(X, y)
        proba = model.predict_proba(np.array([[0.5]]))
        np.testing.assert_allclose(proba, [[2 / 3, 1 / 3]])
        self.assertEqual(model.predict(np.array([[0.5]]))[0], "a")

    def test_distance_weights_exact_match(self):
        X = np.array([[0.0], [1.0], [2.0]])
        model = KNNClassifier(k=3, weights="distance").fit(X, [0, 1, 1])
        np.testing.assert_allclose(model.predict_proba(np.array([[0.0]])), [[1.0, 0.0]])

    def test_regressor_known_answer(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 10.0, 20.0, 30.0])
        model = KNNRegressor(k=2).fit(X, y)
        np.testing.assert_allclose(model.predict(np.array([[0.4], [2.6]])), [5.0, 25.0])
        weighted = KNNRegressor(k=2, weights="distance").fit(X, y)
        self.assertAlmostEqual(weighted.predict(np.array([[0.25]]))[0], 2.5)

    def test_clone_and_pipeline(self):
        model = KNNClassifier(k=7, weights="distance", metric="manhattan")
        self.assertEqual(clone(model).get_params(), model.get_params())
        X, y = circles(200, 0)
        pipe = Pipeline([("knn", model)]).fit(X, y)
        self.assertEqual(pipe.predict_proba(X).shape, (200, 2))


class TestKernels(unittest.TestCase):
    def test_known_values(self):
        a, b = np.array([[1.0, 2.0]]), np.array([[3.0, -1.0]])
        self.assertAlmostEqual(linear_kernel(a, b)[0, 0], 1.0)
        self.assertAlmostEqual(polynomial_kernel(a, b, degree=2, coef0=1.0)[0, 0], 4.0)
        self.assertAlmostEqual(rbf_kernel(a, b, gamma=0.1)[0, 0], np.exp(-1.3))

    def test_gram_matrices_are_psd(self):
        X = np.random.default_rng(0).normal(size=(30, 3))
        for K in [linear_kernel(X, X), rbf_kernel(X, X, 0.5),
                  polynomial_kernel(X, X, 3, 1.0, 1.0)]:
            np.testing.assert_allclose(K, K.T, atol=1e-10)
            self.assertGreater(np.linalg.eigvalsh(K).min(), -1e-6 * np.abs(K).max())
        np.testing.assert_allclose(np.diag(rbf_kernel(X, X, 2.0)), 1.0)


class TestSVM(unittest.TestCase):
    def test_two_point_hard_margin(self):
        # Points at x = -1 and x = +1: w = (1, 0), b = 0, each alpha = 1/2.
        X = np.array([[-1.0, 0.0], [1.0, 0.0]])
        y = np.array([-1.0, 1.0])
        alpha, b, _ = smo(linear_kernel(X, X), y, C=100.0, tol=1e-8)
        np.testing.assert_allclose(alpha, [0.5, 0.5], atol=1e-8)
        self.assertAlmostEqual(b, 0.0, places=8)

    def test_dual_feasibility_and_kkt(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 2))
        y = np.where(X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=120) > 0, 1.0, -1.0)
        C = 1.0
        K = rbf_kernel(X, X, 0.5)
        alpha, b, _ = smo(K, y, C, tol=1e-6)
        self.assertAlmostEqual(alpha @ y, 0.0, places=8)
        self.assertTrue(np.all((alpha >= -1e-12) & (alpha <= C + 1e-12)))
        margin = y * (K @ (alpha * y) + b)
        free = (alpha > 1e-6) & (alpha < C - 1e-6)
        np.testing.assert_allclose(margin[free], 1.0, atol=1e-3)
        self.assertTrue(np.all(margin[alpha < 1e-8] >= 1 - 1e-3))
        self.assertTrue(np.all(margin[alpha > C - 1e-8] <= 1 + 1e-3))

    def test_kernel_matters_on_circles(self):
        X, y = circles(300, 0)
        X_test, y_test = circles(500, 1)
        rbf = KernelSVM(C=1.0, kernel="rbf", gamma=1.0).fit(X, y)
        linear = KernelSVM(C=1.0, kernel="linear").fit(X, y)
        self.assertGreater(np.mean(rbf.predict(X_test) == y_test), 0.9)
        self.assertLess(np.mean(linear.predict(X_test) == y_test), 0.7)
        self.assertLess(len(rbf.support_), len(X))
        self.assertEqual(rbf.decision_function(X_test).shape, (500,))

    def test_labels_and_validation(self):
        X, y = circles(100, 2)
        labels = np.where(y == 1, "in", "out")
        model = KernelSVM(kernel="poly", degree=2).fit(X, labels)
        self.assertTrue(set(model.predict(X)) <= {"in", "out"})
        with self.assertRaises(ValueError):
            KernelSVM().fit(X, np.arange(100) % 3)


if __name__ == "__main__":
    unittest.main()
