"""Chapter 6 lab: Listings 6.12, 6.13, 6.14, 6.15.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch06_evaluation_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 6.12 ----
import numpy as np

from tinyml.base import Estimator, Pipeline
from tinyml.data import make_churn
from tinyml.linear import LogisticRegression, PolynomialFeatures
from tinyml.metrics import accuracy, log_loss, roc_auc
from tinyml.model_selection import (StratifiedKFold, corrected_resampled_ttest,
                                    cross_validate)
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]


class MajorityClass(Estimator):
    """A baseline that ignores X and predicts the training base rate."""

    def __init__(self):  # no hyperparameters, but clone() needs a signature
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.rate_ = np.mean(y == self.classes_[1])
        return self

    def predict_proba(self, X):
        n = len(next(iter(X.values()))) if isinstance(X, dict) else len(X)
        return np.tile([1 - self.rate_, self.rate_], (n, 1))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def logistic(numeric, categorical, degree=1, alpha=1e-3):
    steps = [("encode", TableEncoder(numeric, categorical))]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree)))
    return Pipeline(steps + [("model", LogisticRegression(alpha=alpha))])


table, y = make_churn()
X_train, X_test, y_train, y_test = train_test_split(
    table, y, test_size=0.2, seed=0, stratify=True)

models = {"majority class": MajorityClass(),
          "numeric only": logistic(NUMERIC, []),
          "all features": logistic(NUMERIC, CATEGORICAL),
          "degree 2": logistic(NUMERIC, CATEGORICAL, degree=2, alpha=1e-2)}
scoring = {"accuracy": accuracy, "roc_auc": roc_auc, "log_loss": log_loss}
cv = StratifiedKFold(n_splits=10, seed=0)

results = {}
print(f"{'model':15s} {'accuracy':>15s} {'ROC AUC':>15s} {'log loss':>15s}")
for name, model in models.items():
    results[name] = cross_validate(model, X_train, y_train, cv, scoring)
    cells = [f"{s.mean():.3f} +/- {s.std(ddof=1):.3f}" for s in results[name].values()]
    print(f"{name:15s} " + " ".join(f"{c:>15s}" for c in cells))

# Is degree 2 really better than the plain model? Compare fold by fold.
a, b = results["degree 2"]["roc_auc"], results["all features"]["roc_auc"]
t, p = corrected_resampled_ttest(a, b, test_train_ratio=1 / 9)
print(f"AUC gain of degree 2: {np.mean(a - b):+.4f}, corrected t = {t:.2f}, p = {p:.3f}")


# ---- Listing 6.13 ----
from tinyml.metrics import f1, precision, precision_recall_curve, recall
from tinyml.model_selection import bootstrap_ci, mcnemar, paired_bootstrap

final = logistic(NUMERIC, CATEGORICAL).fit(X_train, y_train)
p_test = final.predict_proba(X_test)[:, 1]
y_hat = final.predict(X_test)

for name, metric, output in [("accuracy", accuracy, y_hat),
                             ("ROC AUC", roc_auc, p_test),
                             ("recall", recall, y_hat)]:
    estimate, low, high = bootstrap_ci(y_test, output, metric, n_boot=2000, seed=0)
    print(f"test {name:9s} {estimate:.3f}  95% CI [{low:.3f}, {high:.3f}]")

simple = logistic(NUMERIC, []).fit(X_train, y_train)
b, c, p = mcnemar(y_test, y_hat, simple.predict(X_test))
print(f"McNemar vs numeric only: {b} vs {c} discordant, p = {p:.3f}")
gain, low, high = paired_bootstrap(y_test, p_test, simple.predict_proba(X_test)[:, 1],
                                   roc_auc, n_boot=2000, seed=0)
print(f"AUC gain over numeric only: {gain:.3f}  95% CI [{low:.3f}, {high:.3f}]")

print("threshold  flagged  precision  recall    F1")
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    flag = (p_test >= threshold).astype(int)
    print(f"{threshold:9.1f}  {flag.mean():7.3f}  {precision(y_test, flag):9.3f}"
          f"  {recall(y_test, flag):6.3f}  {f1(y_test, flag):.3f}")

# The cheapest threshold that still finds 80% of churners:
prec, rec, thresholds = precision_recall_curve(y_test, p_test)
k = np.flatnonzero(rec >= 0.8)[0]
print(f"recall >= 0.8 needs threshold {thresholds[k]:.3f}: "
      f"precision {prec[k]:.3f}, recall {rec[k]:.3f}")


# ---- Listing 6.14 ----
from tinyml.metrics import brier_score, calibration_curve
from tinyml.model_selection import cross_val_predict

# Out-of-fold probabilities give every training row an honest prediction.
cv = StratifiedKFold(n_splits=10, seed=0)
for alpha in [1e-3, 1.0]:
    p = cross_val_predict(logistic(NUMERIC, CATEGORICAL, alpha=alpha),
                          X_train, y_train, cv, method="predict_proba")
    print(f"alpha={alpha:g}: log loss {log_loss(y_train, p):.3f}, "
          f"Brier {brier_score(y_train, p):.3f}")
    print("  mean predicted  observed rate  count")
    for mean_p, observed, count in zip(*calibration_curve(y_train, p, n_bins=5)):
        print(f"  {mean_p:14.3f}  {observed:13.3f}  {count:5d}")


# ---- Listing 6.15 ----
LEAKY = NUMERIC + ["retention_offer_days"]
leaky = Pipeline([("encode", TableEncoder(LEAKY, CATEGORICAL, missing_indicators=True)),
                  ("model", LogisticRegression())])
cv = StratifiedKFold(n_splits=10, seed=0)
scores = cross_validate(leaky, X_train, y_train, cv, {"accuracy": accuracy})
print(f"CV accuracy with retention_offer_days: {scores['accuracy'].mean():.3f}")

# Screen: how well does each single encoded column rank churners by itself?
encoder = TableEncoder(LEAKY, CATEGORICAL, missing_indicators=True).fit(X_train)
X = encoder.transform(X_train)
screen = []
for j, name in enumerate(encoder.feature_names_):
    auc = roc_auc(y_train, X[:, j])
    screen.append((max(auc, 1 - auc), name))  # direction doesn't matter
for auc, name in sorted(screen, reverse=True)[:5]:
    print(f"  single-feature AUC {auc:.3f}  {name}")

has_offer = ~np.isnan(X_train["retention_offer_days"])
print(f"churn rate with an offer recorded {y_train[has_offer].mean():.3f}, "
      f"without {y_train[~has_offer].mean():.3f}")
