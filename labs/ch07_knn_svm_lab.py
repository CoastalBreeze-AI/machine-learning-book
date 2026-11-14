"""Chapter 7 lab: Listings 7.3, 7.4, 7.5, 7.6, 7.10, 7.11, 7.12.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch07_knn_svm_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 7.3 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.metrics import accuracy
from tinyml.neighbors import KNNClassifier
from tinyml.preprocessing import (SimpleImputer, StandardScaler, TableEncoder,
                                  train_test_split)

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
table, y = make_churn()
train, test, y_train, y_test = train_test_split(table, y, test_size=0.2, seed=0,
                                                stratify=True)


def numeric_matrix(t):
    """The numeric columns as a matrix, in their original units."""
    return np.column_stack([t[c] for c in NUMERIC]).astype(float)


imputer = SimpleImputer().fit(numeric_matrix(train))
X_train_raw = imputer.transform(numeric_matrix(train))
X_test_raw = imputer.transform(numeric_matrix(test))
print("standard deviations:", np.round(X_train_raw.std(axis=0), 1))

unscaled = KNNClassifier(k=25).fit(X_train_raw, y_train)
scaled = Pipeline([("scale", StandardScaler()),
                   ("knn", KNNClassifier(k=25))]).fit(X_train_raw, y_train)
full = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                 ("knn", KNNClassifier(k=25))]).fit(train, y_train)
print(f"numeric only, unscaled: {accuracy(y_test, unscaled.predict(X_test_raw)):.3f}")
print(f"numeric only, scaled:   {accuracy(y_test, scaled.predict(X_test_raw)):.3f}")
print(f"all features, scaled:   {accuracy(y_test, full.predict(test)):.3f}")


# ---- Listing 7.4 ----
from tinyml.metrics import roc_auc

# Hold out part of the training set; the test set stays untouched.
fit_part, val_part, y_fit, y_val = train_test_split(train, y_train, test_size=0.25,
                                                    seed=1, stratify=True)
encoder = TableEncoder(NUMERIC, CATEGORICAL).fit(fit_part)
X_fit, X_val = encoder.transform(fit_part), encoder.transform(val_part)

print("    k   train acc   val acc   val AUC")
for k in [1, 3, 9, 25, 51, 101, 201, 501]:
    model = KNNClassifier(k=k).fit(X_fit, y_fit)
    train_acc = accuracy(y_fit, model.predict(X_fit))
    proba = model.predict_proba(X_val)[:, 1]
    print(f"{k:5d}   {train_acc:9.3f}   {accuracy(y_val, proba > 0.5):7.3f}"
          f"   {roc_auc(y_val, proba):7.3f}")


# ---- Listing 7.5 ----
rng = np.random.default_rng(0)
print("    d   nearest/farthest   edge for 1% of the cube")
for d in [1, 2, 10, 100, 1000]:
    points = rng.random((1000, d))  # uniform in the unit cube [0, 1]^d
    query = rng.random(d)
    dist = np.sqrt(((points - query) ** 2).sum(axis=1))
    # A sub-cube holding 1% of the volume has edge 0.01 ** (1 / d).
    print(f"{d:5d}   {dist.min() / dist.max():16.3f}   {0.01 ** (1 / d):10.3f}")


# ---- Listing 7.6 ----
from tinyml.linear import LogisticRegression

encoder = TableEncoder(NUMERIC, CATEGORICAL).fit(train)
X_train, X_test = encoder.transform(train), encoder.transform(test)
print("noise features   kNN acc   logistic acc")
for extra in [0, 10, 50, 200]:
    noise = np.random.default_rng(extra)
    A = np.hstack([X_train, noise.normal(size=(len(X_train), extra))])
    B = np.hstack([X_test, noise.normal(size=(len(X_test), extra))])
    knn = KNNClassifier(k=51).fit(A, y_train)
    logistic = LogisticRegression().fit(A, y_train)
    print(f"{extra:14d}   {accuracy(y_test, knn.predict(B)):7.3f}"
          f"   {accuracy(y_test, logistic.predict(B)):12.3f}")


# ---- Listing 7.10 ----
from tinyml.svm import KernelSVM


def make_circles(n=400, noise=0.1, factor=0.5, seed=0):
    """Two noisy concentric circles: class 1 inside, class 0 outside."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    label = (np.arange(n) % 2).astype(int)
    radius = np.where(label == 1, factor, 1.0)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    return X + rng.normal(0, noise, X.shape), label


X_c, y_c = make_circles(400, seed=0)
X_c_test, y_c_test = make_circles(1000, seed=1)
models = {"linear SVM": KernelSVM(C=1.0, kernel="linear"),
          "poly SVM (degree 2)": KernelSVM(C=1.0, kernel="poly", degree=2),
          "RBF SVM (gamma 1)": KernelSVM(C=1.0, kernel="rbf", gamma=1.0),
          "kNN (k = 15)": KNNClassifier(k=15)}
for name, model in models.items():
    model.fit(X_c, y_c)
    n_sv = len(model.support_) if hasattr(model, "support_") else len(X_c)
    print(f"{name:20s} train {accuracy(y_c, model.predict(X_c)):.3f}"
          f"  test {accuracy(y_c_test, model.predict(X_c_test)):.3f}"
          f"  stored points {n_sv}")


# ---- Listing 7.11 ----
def ascii_map(model, size=21):
    """Prints the predicted class on a grid over [-1.5, 1.5]^2."""
    grid = np.linspace(-1.5, 1.5, size)
    for y_value in grid[::-1]:
        row = np.column_stack([grid, np.full(size, y_value)])
        print("".join("o" if c == 1 else "." for c in model.predict(row)))


ascii_map(models["linear SVM"])
print()
ascii_map(models["RBF SVM (gamma 1)"])


# ---- Listing 7.12 ----
import time

candidates = {
    "logistic regression": LogisticRegression(),
    "kNN (k = 51)": KNNClassifier(k=51),
    "linear SVM (C = 0.1)": KernelSVM(C=0.1, kernel="linear"),
    "RBF SVM (C = 0.1)": KernelSVM(C=0.1, kernel="rbf", gamma=0.05),
    "RBF SVM (C = 1)": KernelSVM(C=1.0, kernel="rbf", gamma=0.05),
}
print("model                    acc    AUC   fit s  predict s  stored rows")
for name, model in candidates.items():
    pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)), ("model", model)])
    start = time.perf_counter()
    pipe.fit(train, y_train)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    X_enc = pipe["encode"].transform(test)
    fitted = pipe["model"]
    score = (fitted.decision_function(X_enc) if hasattr(fitted, "decision_function")
             else fitted.predict_proba(X_enc)[:, 1])
    predict_time = time.perf_counter() - start
    stored = len(getattr(fitted, "support_", getattr(fitted, "X_", [])))
    print(f"{name:22s} {accuracy(y_test, pipe.predict(test)):.3f}  "
          f"{roc_auc(y_test, score):.3f}  {fit_time:6.2f}  {predict_time:9.3f}  {stored:11d}")
