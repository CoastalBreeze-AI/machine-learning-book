"""Chapter 4 lab: Listings 4.12.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch04_linear_regression_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 4.12 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.linear import LinearRegression, add_intercept, mse_objective
from tinyml.optimize import check_gradient, gradient_descent
from tinyml.preprocessing import StandardScaler, train_test_split


def billing_table(table):
    """Features and target for predicting total_charges where it is recorded."""
    known = ~np.isnan(table["total_charges"])
    tenure = table["tenure_months"][known]
    monthly = table["monthly_charges"][known]
    features = {"tenure": tenure, "monthly": monthly, "product": tenure * monthly}
    return features, table["total_charges"][known]


def rmse(model, X, target):
    return np.sqrt(np.mean((model.predict(X) - target) ** 2))


table, y = make_churn()
train, test, _, _ = train_test_split(table, y, test_size=0.2, seed=0, stratify=True)
F_train, t_train = billing_table(train)
F_test, t_test = billing_table(test)

TWO, ALL = ["tenure", "monthly"], ["tenure", "monthly", "product"]
experiments = [
    ("normal, two features, scaled", TWO, True, LinearRegression()),
    ("normal, with product, scaled", ALL, True, LinearRegression()),
    ("gd, with product, scaled", ALL, True,
     LinearRegression("gd", lr=0.3, n_iter=2000)),
    ("normal, with product, raw", ALL, False, LinearRegression()),
    ("gd, with product, raw", ALL, False,
     LinearRegression("gd", lr=1e-7, n_iter=2000)),
]
fitted = {}
for name, columns, scale, model in experiments:
    X_train = np.column_stack([F_train[c] for c in columns])
    X_test = np.column_stack([F_test[c] for c in columns])
    if scale:  # standardize with the training means and standard deviations
        model = Pipeline([("scale", StandardScaler()), ("model", model)])
    fitted[name] = model.fit(X_train, t_train)
    final = model["model"] if scale else model
    weights = " ".join(f"{v:8.2f}" for v in np.r_[final.intercept_, final.coef_])
    print(f"{name:<29} RMSE train {rmse(model, X_train, t_train):5.1f} "
          f"test {rmse(model, X_test, t_test):5.1f}   w {weights}")

A = add_intercept(np.column_stack([F_train[c] for c in ALL]))
w = np.random.default_rng(0).normal(size=4)
print(f"gradient check on the raw problem: relative error "
      f"{check_gradient(mse_objective(A, t_train), w):.1e}")
