"""Chapter 11 lab: Listings 11.17, 11.19, 11.20.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch11_mlp_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 11.17 ----
import numpy as np

from tinyml.autograd import Tensor, check_grad
from tinyml.nn import MLP, cross_entropy

rng = np.random.default_rng(0)
x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
w = Tensor(rng.normal(size=(4, 2)), requires_grad=True)
b = Tensor(rng.normal(size=2), requires_grad=True)

tests = {
    "add, broadcast": lambda v: (v[0] @ v[1] + v[2]).sum(),
    "mul, div, pow": lambda v: ((v[0] * 2.0 / (1.0 + v[0] ** 2)) ** 3).sum(),
    "exp, log": lambda v: (v[0].exp() + 1.0).log().mean(),
    "tanh, sigmoid": lambda v: (v[0].tanh() * v[0].sigmoid()).sum(),
    "relu, max": lambda v: v[0].relu().max(axis=1).sum(),
    "reshape, T, index": lambda v: (v[0].reshape(2, 6).T[1:4] ** 2).sum(),
}
for name, f in tests.items():
    inputs = [x, w, b] if name == "add, broadcast" else [x]
    print(f"{name:18s} max error {check_grad(f, inputs):.1e}")

net = MLP([4, 8, 8, 3], rng=0)
y = np.array([0, 2, 1])
err = check_grad(lambda ps: cross_entropy(net(x), y), net.parameters())
print(f"{'MLP, 4-8-8-3':18s} max error {err:.1e}")


# ---- Listing 11.19 ----
import numpy as np

from tinyml.metrics import accuracy
from tinyml.nn import MLPClassifier


def make_spirals(n=600, noise=0.2, seed=0):
    """Two interleaved spirals in the plane: a classic nonlinear problem."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    t = rng.uniform(0.5, 3.0 * np.pi, n)        # angle along the spiral
    X = np.column_stack([t * np.cos(t + np.pi * y), t * np.sin(t + np.pi * y)])
    return X / 5.0 + rng.normal(0, noise, (n, 2)) / 5.0, y


X_train, y_train = make_spirals(600, seed=0)
X_test, y_test = make_spirals(600, seed=1)

for hidden in [(), (16,), (64, 64)]:
    model = MLPClassifier(hidden=hidden, lr=0.01, epochs=300, seed=0)
    model.fit(X_train, y_train)
    print(f"hidden={str(hidden):9s} train {accuracy(y_train, model.predict(X_train)):.3f}"
          f"  test {accuracy(y_test, model.predict(X_test)):.3f}"
          f"  final loss {model.loss_curve_[-1]:.3f}")


# ---- Listing 11.20 ----
import numpy as np

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.linear import LogisticRegression
from tinyml.metrics import accuracy, roc_auc
from tinyml.nn import MLPClassifier
from tinyml.preprocessing import TableEncoder, train_test_split

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]

table, y = make_churn(seed=0)
train, test, y_train, y_test = train_test_split(table, y, test_size=0.2, seed=0,
                                                stratify=True)
models = {
    "logistic regression": LogisticRegression(),
    "MLP, no hidden layer": MLPClassifier(hidden=(), epochs=30),
    "MLP, 32 hidden": MLPClassifier(hidden=(32,), epochs=30),
    "MLP, 64-64 hidden": MLPClassifier(hidden=(64, 64), epochs=30),
    "MLP, 64-64, decay": MLPClassifier(hidden=(64, 64), epochs=30, weight_decay=1e-3),
}
for name, model in models.items():
    pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)), ("model", model)])
    pipe.fit(train, y_train)
    p_train = pipe.predict_proba(train)[:, 1]
    p_test = pipe.predict_proba(test)[:, 1]
    print(f"{name:21s} train acc {accuracy(y_train, p_train >= 0.5):.3f}"
          f"  test acc {accuracy(y_test, p_test >= 0.5):.3f}"
          f"  test AUC {roc_auc(y_test, p_test):.3f}")
