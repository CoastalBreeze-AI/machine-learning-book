"""Tests for Chapter 10: tinyml.cluster and tinyml.decomposition."""
import unittest

import numpy as np

from tinyml.base import clone
from tinyml.cluster import (GaussianMixture, KMeans, adjusted_rand_index, kmeans_plus_plus,
                            logsumexp, silhouette_samples, silhouette_score, squared_distances)
from tinyml.decomposition import PCA


def blobs(seed=0, n=100):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.vstack([rng.normal(c, 1.0, (n, 2)) for c in centers])
    return X, np.repeat(np.arange(3), n)


class TestHelpers(unittest.TestCase):
    def test_squared_distances(self):
        rng = np.random.default_rng(0)
        X, C = rng.normal(size=(7, 3)), rng.normal(size=(4, 3))
        brute = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_allclose(squared_distances(X, C), brute, atol=1e-12)

    def test_logsumexp(self):
        a = np.array([[1000.0, 1000.0], [-1000.0, -1001.0]])
        np.testing.assert_allclose(logsumexp(a, axis=1),
                                   [1000 + np.log(2), -1000 + np.log1p(np.exp(-1))])

    def test_kmeans_plus_plus_picks_data_points(self):
        X, _ = blobs()
        centers = kmeans_plus_plus(X, 3, np.random.default_rng(0))
        self.assertEqual(centers.shape, (3, 2))
        for c in centers:
            self.assertTrue(np.any(np.all(X == c, axis=1)))


class TestKMeans(unittest.TestCase):
    def test_recovers_separated_blobs(self):
        X, truth = blobs()
        model = KMeans(n_clusters=3, seed=0).fit(X)
        self.assertAlmostEqual(adjusted_rand_index(truth, model.labels_), 1.0)
        recomputed = sum(np.sum((X[model.labels_ == j] - c) ** 2)
                         for j, c in enumerate(model.cluster_centers_))
        self.assertAlmostEqual(model.inertia_, recomputed, places=6)
        np.testing.assert_array_equal(model.predict(X), model.labels_)
        self.assertEqual(model.transform(X).shape, (300, 3))

    def test_more_starts_never_worse(self):
        X, _ = blobs(seed=3)
        one = KMeans(n_clusters=6, n_init=1, seed=0).fit(X).inertia_
        many = KMeans(n_clusters=6, n_init=10, seed=0).fit(X).inertia_
        self.assertLessEqual(many, one + 1e-9)

    def test_clone_and_params(self):
        model = clone(KMeans(n_clusters=4, n_init=2, seed=5))
        self.assertEqual(model.get_params()["n_clusters"], 4)


class TestGaussianMixture(unittest.TestCase):
    def test_likelihood_never_decreases(self):
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(0, 1, (150, 2)), rng.normal(2.0, 0.5, (150, 2))])
        for cov in ["full", "diag"]:
            model = GaussianMixture(n_components=3, covariance_type=cov, n_iter=50,
                                    tol=0.0, seed=0).fit(X)
            history = np.array(model.log_likelihood_history_)
            self.assertTrue(np.all(np.diff(history) >= -1e-10), cov)
            proba = model.predict_proba(X)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0)
            self.assertAlmostEqual(model.score(X), history[-1])

    def test_single_component_is_mle_gaussian(self):
        rng = np.random.default_rng(2)
        X = rng.multivariate_normal([1, -1], [[2.0, 0.6], [0.6, 1.0]], size=500)
        model = GaussianMixture(n_components=1, reg_covar=0.0).fit(X)
        mean, cov = X.mean(axis=0), np.cov(X.T, bias=True)
        np.testing.assert_allclose(model.means_[0], mean)
        np.testing.assert_allclose(model.covariances_[0], cov)
        diff = X - mean
        direct = -0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(cov))
                         + np.sum(diff @ np.linalg.inv(cov) * diff, axis=1))
        np.testing.assert_allclose(model.score_samples(X), direct)

    def test_recovers_blobs(self):
        X, truth = blobs()
        model = GaussianMixture(n_components=3, seed=0).fit(X)
        self.assertAlmostEqual(adjusted_rand_index(truth, model.predict(X)), 1.0)
        self.assertLess(model.bic(X), GaussianMixture(n_components=1).fit(X).bic(X))


class TestEvaluation(unittest.TestCase):
    def test_silhouette_known_answer(self):
        X = np.array([[0.0], [1.0], [10.0], [11.0]])
        labels = np.array([0, 0, 1, 1])
        expected = np.array([(10.5 - 1) / 10.5, (9.5 - 1) / 9.5, (9.5 - 1) / 9.5, (10.5 - 1) / 10.5])
        np.testing.assert_allclose(silhouette_samples(X, labels), expected)
        self.assertAlmostEqual(silhouette_score(X, labels), expected.mean())

    def test_silhouette_singleton_is_zero(self):
        X = np.array([[0.0], [1.0], [5.0]])
        self.assertEqual(silhouette_samples(X, np.array([0, 0, 1]))[2], 0.0)

    def test_adjusted_rand_index(self):
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]), 4 / 7)
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [5, 5, 3, 3]), 1.0)
        rng = np.random.default_rng(0)
        self.assertLess(abs(adjusted_rand_index(rng.integers(0, 3, 3000),
                                                rng.integers(0, 3, 3000))), 0.02)


class TestPCA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 4)) + 3.0

    def test_matches_covariance_eigendecomposition(self):
        pca = PCA().fit(self.X)
        values, vectors = np.linalg.eigh(np.cov(self.X.T))
        np.testing.assert_allclose(pca.explained_variance_, values[::-1])
        for component, vector in zip(pca.components_, vectors.T[::-1]):
            self.assertAlmostEqual(abs(component @ vector), 1.0)
        np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(4), atol=1e-12)
        self.assertAlmostEqual(pca.explained_variance_ratio_.sum(), 1.0)

    def test_round_trip_and_truncation(self):
        full = PCA().fit(self.X)
        np.testing.assert_allclose(full.inverse_transform(full.transform(self.X)), self.X)
        two = PCA(n_components=2).fit(self.X)
        Z = two.transform(self.X)
        self.assertEqual(Z.shape, (200, 2))
        np.testing.assert_allclose(Z.var(axis=0, ddof=1), two.explained_variance_)
        # Eckart-Young: the squared error equals the discarded variance.
        error = np.sum((self.X - two.inverse_transform(Z)) ** 2)
        self.assertAlmostEqual(error, np.sum(full.explained_variance_[2:]) * 199, places=6)

    def test_sign_convention(self):
        pca = PCA().fit(self.X)
        for component in pca.components_:
            self.assertGreater(component[np.abs(component).argmax()], 0)


if __name__ == "__main__":
    unittest.main()
