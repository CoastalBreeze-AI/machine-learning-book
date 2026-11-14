"""Chapter 16 lab: Listings 16.5, 16.6, 16.7, 16.8.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch16_audit_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 16.5 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.fairness import (demographic_parity_difference, disparate_impact_ratio,
                             equalized_odds_difference, group_rates)
from tinyml.linear import LogisticRegression
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]

table, y = make_churn(n=5000, seed=0)
X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                    seed=0, stratify=True)
audit_table, y_audit = make_churn(n=20000, seed=7)  # a large, fresh audit sample


def age_group(t):
    return np.where(t["age"] >= 65, "65+", "under 65")


def fit(numeric):
    return Pipeline([("encode", TableEncoder(numeric, CATEGORICAL)),
                     ("classify", LogisticRegression())]).fit(X_train, y_train)


def audit(name, model, t, y_true, threshold=0.5, show_groups=False):
    flagged = (model.predict_proba(t)[:, 1] >= threshold).astype(int)
    group = age_group(t)
    print(f"{name}: DP diff {demographic_parity_difference(flagged, group):.3f}, "
          f"DI ratio {disparate_impact_ratio(flagged, group):.2f}, "
          f"EO diff {equalized_odds_difference(y_true, flagged, group):.3f}")
    if show_groups:
        print(f"  {'group':<10}{'n':>6}{'base':>7}{'select':>8}{'TPR':>7}"
              f"{'FPR':>7}{'prec':>7}")
        for g, r in group_rates(y_true, flagged, group).items():
            print(f"  {g:<10}{r['n']:>6}{r['base_rate']:>7.3f}{r['selection_rate']:>8.3f}"
                  f"{r['tpr']:>7.3f}{r['fpr']:>7.3f}{r['precision']:>7.3f}")


model = fit(NUMERIC)
audit("test set (1,000 rows)", model, X_test, y_test, show_groups=True)
audit("audit sample (20,000 rows)", model, audit_table, y_audit, show_groups=True)
blind = fit([c for c in NUMERIC if c != "age"])  # "fairness through unawareness"
audit("without age, audit sample", blind, audit_table, y_audit)
calls, group = audit_table["support_calls"], age_group(audit_table)
print(f"mean support calls: 65+ {calls[group == '65+'].mean():.2f}, "
      f"under 65 {calls[group == 'under 65'].mean():.2f}")


# ---- Listing 16.6 ----
from tinyml.fairness import group_calibration, thresholds_for_tpr
from tinyml.metrics import accuracy

group = age_group(audit_table)
for name, m in [("with age", model), ("without age", blind)]:
    score = m.predict_proba(audit_table)[:, 1]
    senior = group == "65+"
    print(f"{name}: 65+ mean score {score[senior].mean():.3f} "
          f"vs. churn rate {y_audit[senior].mean():.3f}")

score = model.predict_proba(audit_table)[:, 1]
print("calibration, (mean score, churn rate, n) per score bin:")
for g, bins in group_calibration(y_audit, score, group).items():
    print(f"  {g:<9}", "  ".join(f"({s:.2f}, {r:.2f}, {n})" for s, r, n in bins))

cut = thresholds_for_tpr(y_audit, score, group, target_tpr=0.54)
print("per-group thresholds for TPR 0.54:", {g: round(t, 3) for g, t in cut.items()})
threshold = np.where(group == "65+", cut["65+"], cut["under 65"])
flagged = (score >= threshold).astype(int)
for g, r in group_rates(y_audit, flagged, group).items():
    print(f"  {g:<9} select {r['selection_rate']:.3f}  TPR {r['tpr']:.3f}  "
          f"FPR {r['fpr']:.3f}  precision {r['precision']:.3f}")
print(f"accuracy: one threshold {accuracy(y_audit, (score >= 0.5).astype(int)):.4f}, "
      f"per-group thresholds {accuracy(y_audit, flagged):.4f}")


# ---- Listing 16.7 ----
from tinyml.explain import partial_dependence, permutation_importance

RAW = NUMERIC + CATEGORICAL
print("Permutation importance (drop in mean log-likelihood), held-out test set:")
for name, (mean, std) in permutation_importance(model, X_test, y_test, RAW,
                                                n_repeats=10).items():
    print(f"  {name:<16}{mean:>8.4f} +/- {std:.4f}")

leaky = Pipeline([("encode", TableEncoder(NUMERIC + ["retention_offer_days"],
                                          CATEGORICAL, missing_indicators=True)),
                  ("classify", LogisticRegression())]).fit(X_train, y_train)
top = permutation_importance(leaky, X_test, y_test, RAW + ["retention_offer_days"],
                             n_repeats=10)
print("With the leaky column, top three:",
      [(name, round(mean, 3)) for name, (mean, _) in list(top.items())[:3]])

for column in ["contract", "support_calls"]:
    grid, average = partial_dependence(model, X_test, column)
    print(f"Partial dependence on {column}:",
          ", ".join(f"{g}: {a:.2f}" for g, a in zip(grid, average)))


# ---- Listing 16.8 ----
from tinyml.privacy import dp_mean

charges = X_train["monthly_charges"]  # 4,000 customers
true_mean = charges.mean()
rng = np.random.default_rng(0)
print(f"true mean monthly charge: {true_mean:.3f}")
print(f"{'epsilon':>8}{'RMSE':>10}{'95% of errors within':>24}")
for epsilon in [0.01, 0.1, 1.0, 10.0]:
    errors = np.array([dp_mean(charges, 0, 150, epsilon, rng) - true_mean
                       for _ in range(2000)])
    print(f"{epsilon:>8}{np.sqrt(np.mean(errors ** 2)):>10.3f}"
          f"{np.quantile(np.abs(errors), 0.95):>24.3f}")

print("\nClipping bounds at epsilon = 0.1 (bias from clipping vs. noise):")
for upper in [60, 100, 150, 1000]:
    estimates = np.array([dp_mean(charges, 0, upper, 0.1, rng) for _ in range(2000)])
    bias = estimates.mean() - true_mean
    print(f"  upper {upper:>5}: bias {bias:>7.3f}, std {estimates.std():>7.3f}")
