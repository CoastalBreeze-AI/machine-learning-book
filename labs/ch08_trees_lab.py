"""Chapter 8 lab: Listings 8.5, 8.6, 8.7, 8.9, 8.12, 8.13.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch08_trees_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 8.5 ----
import numpy as np

from tinyml.data import make_churn
from tinyml.preprocessing import train_test_split
from tinyml.tree import DecisionTreeClassifier, export_text

table, y = make_churn()
train, test, y_train, y_test = train_test_split(table, y, test_size=0.2, seed=0,
                                                stratify=True)
NAMES = ["tenure_months", "monthly_charges", "support_calls", "age",
         "month_to_month", "one_year", "electronic"]


def readable(t):
    """Features in their original units; trees don't need scaling."""
    return np.column_stack([
        t["tenure_months"], t["monthly_charges"], t["support_calls"], t["age"],
        t["contract"] == "month-to-month", t["contract"] == "one-year",
        t["payment_method"] == "electronic"]).astype(float)


tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)
tree.fit(readable(train), y_train)
print(export_text(tree, NAMES))
print("importances:", {n: round(float(v), 2) for n, v in zip(NAMES, tree.feature_importances_)})


# ---- Listing 8.6 ----
big, y_big = make_churn(n=50_000, seed=1)
calls, m2m = big["support_calls"], big["contract"] == "month-to-month"
print("calls   churn rate (other contracts)   churn rate (month-to-month)")
for c in range(7):
    rates = [y_big[(calls == c) & (m2m == g)].mean() for g in (False, True)]
    print(f"{c:5d}   {rates[0]:28.2f}   {rates[1]:26.2f}")

two = np.column_stack([calls, m2m]).astype(float)
tree2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=200).fit(two, y_big)
print(export_text(tree2, ["support_calls", "month_to_month"]))


# ---- Listing 8.7 ----
X_tr, X_te = readable(train), readable(test)
rng = np.random.default_rng(0)
predictions = []
for b in range(5):
    rows = rng.integers(0, len(X_tr), len(X_tr))  # a bootstrap sample
    t = DecisionTreeClassifier(max_depth=3).fit(X_tr[rows], y_train[rows])
    left = t.left_[0]
    print(f"tree {b}: root {NAMES[t.feature_[0]]} <= {t.threshold_[0]:.1f}, "
          f"then {NAMES[t.feature_[left]]} <= {t.threshold_[left]:.1f}")
    predictions.append(DecisionTreeClassifier(min_samples_leaf=1)
                       .fit(X_tr[rows], y_train[rows]).predict(X_te))
predictions = np.array(predictions)
disagree = np.mean(predictions.min(axis=0) != predictions.max(axis=0))
print(f"full-depth trees disagree on {disagree:.1%} of test customers")


# ---- Listing 8.9 ----
from tinyml.ensemble import RandomForestClassifier
from tinyml.metrics import accuracy, log_loss, roc_auc

print("trees   bagging acc  AUC    forest acc  AUC    forest OOB acc")
for n_trees in [1, 5, 25, 100]:
    row = f"{n_trees:5d}"
    for max_features in [None, "sqrt"]:
        model = RandomForestClassifier(n_trees, max_features, min_samples_leaf=5,
                                       seed=0).fit(X_tr, y_train)
        proba = model.predict_proba(X_te)[:, 1]
        row += f"   {accuracy(y_test, model.predict(X_te)):10.3f}  {roc_auc(y_test, proba):.3f}"
    print(row + f"   {model.oob_score_:14.3f}")


# ---- Listing 8.12 ----
import time

from tinyml.base import Pipeline
from tinyml.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from tinyml.linear import LogisticRegression, sigmoid
from tinyml.model_selection import paired_bootstrap
from tinyml.preprocessing import TableEncoder

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
models = {
    "logistic regression": LogisticRegression(),
    "tree (depth 5)": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20),
    "random forest": RandomForestClassifier(200, "sqrt", min_samples_leaf=5),
    "AdaBoost (stumps)": AdaBoostClassifier(200),
    "gradient boosting": GradientBoostingClassifier(300, learning_rate=0.05,
                                                    max_depth=3, subsample=0.8),
}
scores = {}
print("model                  acc    AUC    log loss  fit s")
for name, model in models.items():
    pipe = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)), ("model", model)])
    start = time.perf_counter()
    pipe.fit(train, y_train)
    seconds = time.perf_counter() - start
    scores[name] = proba = pipe.predict_proba(test)[:, 1]
    print(f"{name:21s}  {accuracy(y_test, proba > 0.5):.3f}  {roc_auc(y_test, proba):.3f}"
          f"  {log_loss(y_test, proba):8.3f}  {seconds:5.1f}")
diff, low, high = paired_bootstrap(y_test, scores["gradient boosting"],
                                   scores["logistic regression"], roc_auc)
print(f"AUC, boosting minus logistic: {diff:+.3f} (95% CI {low:+.3f} to {high:+.3f})")


# ---- Listing 8.13 ----
fit_part, val_part, y_fit, y_val = train_test_split(train, y_train, test_size=0.25,
                                                    seed=1, stratify=True)
encoder = TableEncoder(NUMERIC, CATEGORICAL).fit(fit_part)
X_fit, X_val = encoder.transform(fit_part), encoder.transform(val_part)
for rate in [0.5, 0.05]:
    gb = GradientBoostingClassifier(500, learning_rate=rate, max_depth=3,
                                    subsample=0.8).fit(X_fit, y_fit)
    val_loss = [log_loss(y_val, sigmoid(F))
                for F in gb.staged_decision_function(X_val)]
    best = int(np.argmin(val_loss))
    marks = "  ".join(f"{m}: {val_loss[m - 1]:.3f}" for m in [10, 50, 100, 300, 500])
    print(f"rate {rate}: validation log loss  {marks}")
    print(f"          best {val_loss[best]:.3f} at {best + 1} trees; "
          f"training loss at 500: {gb.train_loss_[-1]:.3f}")
