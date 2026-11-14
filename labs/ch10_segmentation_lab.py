"""Chapter 10 lab: Listings 10.10, 10.11, 10.12.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch10_segmentation_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 10.10 ----
import numpy as np

from tinyml.cluster import (GaussianMixture, KMeans, adjusted_rand_index,
                            silhouette_score)


def make_blobs(seed=0):
    """Four 2-D clusters of different sizes and shapes, with known labels."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [6.0, 6.0]])
    sizes = [200, 150, 100, 150]
    shapes = [np.eye(2), np.eye(2) * 0.5,
              np.array([[2.5, 0.0], [0.0, 0.3]]),           # a wide, flat cluster
              np.array([[1.0, 0.8], [0.8, 1.0]])]           # a tilted cluster
    X = np.vstack([rng.multivariate_normal(c, s, size=m)
                   for c, s, m in zip(centers, shapes, sizes)])
    labels = np.repeat(np.arange(4), sizes)
    return X, labels


X, truth = make_blobs()
print(f"{'k':>2} {'inertia':>9} {'silhouette':>11} {'ARI':>6}")
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, seed=0).fit(X)
    sil = silhouette_score(X, km.labels_) if k > 1 else float("nan")
    print(f"{k:>2} {km.inertia_:9.1f} {sil:11.3f} "
          f"{adjusted_rand_index(truth, km.labels_):6.3f}")

for cov in ["full", "diag"]:
    gmm = GaussianMixture(n_components=4, covariance_type=cov, seed=0).fit(X)
    print(f"GMM {cov}: ARI {adjusted_rand_index(truth, gmm.predict(X)):.3f}, "
          f"BIC {gmm.bic(X):.1f}, {gmm.n_iter_} iterations")
    print("  mean log-likelihood:", np.round(gmm.log_likelihood_history_, 4))


# ---- Listing 10.11 ----
import numpy as np

from tinyml.cluster import KMeans, silhouette_score
from tinyml.data import make_churn
from tinyml.decomposition import PCA
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]

table, y = make_churn()
X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                    seed=0, stratify=True)
encoder = TableEncoder(NUMERIC, ["contract"]).fit(X_train)   # see the text
Z = encoder.transform(X_train)                               # labels not used

pca = PCA().fit(Z)
print("features:", encoder.feature_names_)
print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))

print(f"{'k':>2} {'inertia':>9} {'silhouette':>11}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, seed=0).fit(Z)
    print(f"{k:>2} {km.inertia_:9.1f} "
          f"{silhouette_score(Z, km.labels_, sample_size=2000, seed=0):11.3f}")


# ---- Listing 10.12 ----
# Continues Listing 10.11. Churn labels are used only now, after clustering.
km = KMeans(n_clusters=4, n_init=10, seed=0).fit(Z)
test_segment = km.predict(encoder.transform(X_test))

print(f"{'seg':>3} {'size':>5} {'tenure':>7} {'monthly':>8} {'calls':>6} "
      f"{'age':>5} {'m2m':>5} {'churn':>6} {'test churn':>11}")
for s in range(4):
    rows = km.labels_ == s
    m2m = np.mean(X_train["contract"][rows] == "month-to-month")
    print(f"{s:>3} {rows.sum():>5} {X_train['tenure_months'][rows].mean():7.1f} "
          f"{X_train['monthly_charges'][rows].mean():8.1f} "
          f"{X_train['support_calls'][rows].mean():6.2f} "
          f"{X_train['age'][rows].mean():5.1f} {m2m:5.2f} "
          f"{y_train[rows].mean():6.3f} {y_test[test_segment == s].mean():11.3f}")
print(f"overall churn: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
