"""Chapter 9 lab: Listings 9.7, 9.8, 9.9, 9.10.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch09_naive_bayes_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 9.7 ----
import numpy as np

from tinyml.bayes import BayesianLinearRegression


def cubic_features(x):
    return np.column_stack([x, x ** 2, x ** 3])


def true_f(x):
    return 0.5 * x - 0.8 * x ** 3


rng = np.random.default_rng(0)
noise_sd = 0.1
x_train = np.concatenate([rng.uniform(-1.0, -0.3, 10), rng.uniform(0.3, 1.0, 10)])
y_train = true_f(x_train) + rng.normal(0, noise_sd, len(x_train))

model = BayesianLinearRegression(alpha=1.0, beta=1 / noise_sd ** 2)
model.fit(cubic_features(x_train), y_train)

x_test = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
mean, std = model.predict(cubic_features(x_test), return_std=True)
epistemic = np.sqrt(std ** 2 - 1 / model.beta)       # remove the noise part

print("posterior mean weights:", np.round(model.mean_, 3))
print(f"{'x':>5} {'true f':>8} {'mean':>8} {'std':>7} {'epistemic':>10}")
for row in zip(x_test, true_f(x_test), mean, std, epistemic):
    print("{:5.1f} {:8.3f} {:8.3f} {:7.3f} {:10.3f}".format(*row))


# ---- Listing 9.8 ----
import numpy as np

from tinyml.bayes import BayesianLinearRegression


def cubic_features(x):
    return np.column_stack([x, x ** 2, x ** 3])


def true_f(x):
    return 0.5 * x - 0.8 * x ** 3


rng = np.random.default_rng(1)
noise_sd = 0.1
print(f"{'n':>6} {'std at 0.5':>11} {'epistemic':>10} {'std at 2.0':>11} {'epistemic':>10}")
for n in [5, 20, 200, 2000]:
    x = rng.uniform(-1.0, 1.0, n)
    y = true_f(x) + rng.normal(0, noise_sd, n)
    model = BayesianLinearRegression(alpha=1.0, beta=1 / noise_sd ** 2)
    model.fit(cubic_features(x), y)
    _, std = model.predict(cubic_features(np.array([0.5, 2.0])), return_std=True)
    epi = np.sqrt(std ** 2 - 1 / model.beta)
    print(f"{n:>6} {std[0]:11.3f} {epi[0]:10.3f} {std[1]:11.3f} {epi[1]:10.3f}")


# ---- Listing 9.9 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.linear import LogisticRegression
from tinyml.metrics import accuracy, brier_score, calibration_curve, log_loss, roc_auc
from tinyml.naive_bayes import GaussianNB, MixedNB
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
WITHOUT_TOTAL = [c for c in NUMERIC if c != "total_charges"]

table, y = make_churn()
X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                    seed=0, stratify=True)


def expected_calibration_error(y_true, proba, n_bins=10):
    mean_predicted, fraction_positive, counts = calibration_curve(y_true, proba, n_bins)
    return np.sum(counts * np.abs(mean_predicted - fraction_positive)) / counts.sum()


models = {
    "logistic regression": Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                                     ("model", LogisticRegression())]),
    "Gaussian NB, encoded": Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                                      ("model", GaussianNB())]),
    "mixed NB": MixedNB(NUMERIC, CATEGORICAL),
    "mixed NB, no total": MixedNB(WITHOUT_TOTAL, CATEGORICAL),
}
print(f"{'model':<22} {'acc':>6} {'AUC':>6} {'logloss':>8} {'Brier':>6} {'ECE':>6} {'extreme':>8}")
probabilities = {}
for name, model in models.items():
    p = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    probabilities[name] = p
    extreme = np.mean((p < 0.05) | (p > 0.95))     # share of very confident predictions
    print(f"{name:<22} {accuracy(y_test, (p >= 0.5).astype(int)):6.3f} "
          f"{roc_auc(y_test, p):6.3f} {log_loss(y_test, p):8.3f} "
          f"{brier_score(y_test, p):6.3f} {expected_calibration_error(y_test, p):6.3f} "
          f"{extreme:8.3f}")


# ---- Listing 9.10 ----
# Continues Listing 9.9: probabilities, X_train, y_train, and y_test are defined there.
print(f"{'bin':<9} {'LR mean p':>9} {'observed':>9} {'n':>4}   "
      f"{'NB mean p':>9} {'observed':>9} {'n':>4}")
lr = calibration_curve(y_test, probabilities["logistic regression"], n_bins=5)
nb = calibration_curve(y_test, probabilities["mixed NB"], n_bins=5)
for b in range(5):
    print(f"{b / 5:.1f}-{(b + 1) / 5:.1f}   {lr[0][b]:9.3f} {lr[1][b]:9.3f} {lr[2][b]:4d}   "
          f"{nb[0][b]:9.3f} {nb[1][b]:9.3f} {nb[2][b]:4d}")

# The independence assumption, checked: correlation within each class.
for label in (0, 1):
    rows = (y_train == label) & ~np.isnan(X_train["total_charges"])
    r = np.corrcoef(X_train["tenure_months"][rows], X_train["total_charges"][rows])[0, 1]
    print(f"class {label}: corr(tenure_months, total_charges) = {r:.3f}")
