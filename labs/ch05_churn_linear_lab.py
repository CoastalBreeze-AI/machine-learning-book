"""Chapter 5 lab: Listings 5.7, 5.8, 5.9.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch05_churn_linear_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 5.7 ----
import numpy as np

from tinyml.data import make_churn
from tinyml.linear import Lasso, Ridge
from tinyml.preprocessing import TableEncoder, train_test_split

# Predict support_calls from everything else. In data.py its true mean is
# 0.8 + 0.015 * (monthly - 60) + 0.6 * (age >= 65) + 0.2 * (region == south).
FEATURES = ["tenure_months", "monthly_charges", "total_charges", "age"]
CATEGORIES = ["contract", "payment_method", "region"]

table, y = make_churn()
train, test, _, _ = train_test_split(table, y, test_size=0.2, seed=0, stratify=True)
encoder = TableEncoder(FEATURES, CATEGORIES).fit(train)
X_train, X_test = encoder.transform(train), encoder.transform(test)
calls_train, calls_test = train["support_calls"], test["support_calls"]
names = encoder.feature_names_

alphas = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
paths = [Lasso(alpha).fit(X_train, calls_train) for alpha in alphas]

print(f"{'feature':28s}" + "".join(f"{a:>7g}" for a in alphas))
for j, name in enumerate(names):
    cells = [f"{m.coef_[j]:7.3f}" if m.coef_[j] != 0 else "      ." for m in paths]
    print(f"{name:28s}" + "".join(cells))
for label, model in [("lasso 0.01", paths[3]), ("ridge 1.0", Ridge(1.0))]:
    model.fit(X_train, calls_train)
    test_mse = np.mean((calls_test - model.predict(X_test)) ** 2)
    print(f"{label}: nonzero={np.sum(model.coef_ != 0)}, test MSE={test_mse:.3f}")


# ---- Listing 5.8 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.linear import LogisticRegression
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
# Per-unit log-odds effects from the true model in data.py (main effects).
TRUE = {"tenure_months": -0.045, "monthly_charges": 0.03, "total_charges": 0.0,
        "support_calls": 0.35, "age": -0.01}

table, y = make_churn()
X_train, X_test, y_train, y_test = train_test_split(
    table, y, test_size=0.2, seed=0, stratify=True)
pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                 ("model", LogisticRegression(alpha=1e-3))]).fit(X_train, y_train)
encoder, model = pipe["encode"], pipe["model"]
print(f"test accuracy {np.mean(pipe.predict(X_test) == y_test):.3f}, "
      f"Newton iterations {model.n_iter_}")

# Scaled coefficients are per standard deviation; divide by it for per unit.
per_unit = model.coef_[:len(NUMERIC)] / encoder.scaler_.scale_
print(f"{'feature':16s} {'per SD':>7s} {'per unit':>9s} {'true':>7s} {'odds x':>7s}")
for name, sd_coef, unit_coef in zip(NUMERIC, model.coef_, per_unit):
    print(f"{name:16s} {sd_coef:7.3f} {unit_coef:9.4f} {TRUE[name]:7.3f} "
          f"{np.exp(unit_coef):7.3f}")

# One-hot coefficients are identified only up to a constant per variable,
# so compare differences between categories of the same variable.
coef = dict(zip(encoder.feature_names_, model.coef_))
contrasts = [("contract=month-to-month", "contract=two-year", 1.5),
             ("contract=one-year", "contract=two-year", 0.3),
             ("payment_method=electronic", "payment_method=mailed_check", 0.5),
             ("region=south", "region=north", 0.0)]
for a, b, truth in contrasts:
    print(f"{a} vs {b.split('=')[1]}: {coef[a] - coef[b]:6.3f} (true {truth})")


# ---- Listing 5.9 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.linear import SoftmaxRegression
from tinyml.preprocessing import TableEncoder, train_test_split

# A three-class problem: which contract did the customer choose?
table, _ = make_churn()
contract = table["contract"]
X_train, X_test, c_train, c_test = train_test_split(
    table, contract, test_size=0.2, seed=0, stratify=True)

pipe = Pipeline([("encode", TableEncoder(["tenure_months", "age"], [])),
                 ("model", SoftmaxRegression(alpha=1e-3))]).fit(X_train, c_train)
model = pipe["model"]
values, counts = np.unique(c_train, return_counts=True)
print("classes:", [str(c) for c in model.classes_], "iterations:", model.n_iter_)
print(f"majority-class accuracy {np.mean(c_test == values[counts.argmax()]):.3f}")
print(f"softmax accuracy        {np.mean(pipe.predict(X_test) == c_test):.3f}")
print("coefficients (rows: tenure, age; columns: classes)")
print(np.round(model.coef_, 3))

new = {"tenure_months": np.array([2.0, 24.0, 60.0]), "age": np.array([30.0, 45.0, 70.0])}
for row in np.round(pipe.predict_proba(new), 3):
    print(row)
