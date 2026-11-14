"""Chapter 3 lab: Listings 3.15.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch03_leakage_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 3.15 ----
import numpy as np

from tinyml.base import Estimator, Pipeline
from tinyml.data import make_churn
from tinyml.preprocessing import TableEncoder, train_test_split


class TinyLogistic(Estimator):
    """A bare-bones logistic regression, fitted by gradient descent.

    Chapter 4 explains gradient descent, and Chapter 5 builds a complete
    LogisticRegression. This stand-in exists only for the leakage experiment.
    """

    def __init__(self, lr=0.5, n_iter=500):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])  # ones for the intercept
        self.w_ = np.zeros(A.shape[1])
        for _ in range(self.n_iter):
            p = 1.0 / (1.0 + np.exp(-A @ self.w_))  # predicted P(churn)
            self.w_ -= self.lr * A.T @ (p - y) / len(y)  # step down the gradient
        return self

    def predict(self, X):
        A = np.column_stack([np.ones(len(X)), X])
        return (A @ self.w_ > 0).astype(int)


NUMERIC = ["tenure_months", "monthly_charges", "total_charges",
           "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
LEAKY = NUMERIC + ["retention_offer_days"]

table, y = make_churn()
train, test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                seed=0, stratify=True)
honest = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                   ("model", TinyLogistic())]).fit(train, y_train)
leaky = Pipeline([("encode", TableEncoder(LEAKY, CATEGORICAL,
                                          missing_indicators=True)),
                  ("model", TinyLogistic())]).fit(train, y_train)

# When the model is used, nobody has been offered a retention deal yet.
deployed = dict(test, retention_offer_days=np.full(len(y_test), np.nan))
offer = ~np.isnan(test["retention_offer_days"])
results = [("always predict 'stays'", np.zeros(len(y_test))),
           ("rule: offer recorded => churns", offer),
           ("honest pipeline", honest.predict(test)),
           ("leaky pipeline", leaky.predict(test)),
           ("leaky pipeline, as deployed", leaky.predict(deployed))]
for name, prediction in results:
    print(f"{name:<32} test accuracy {np.mean(prediction == y_test):.3f}")
