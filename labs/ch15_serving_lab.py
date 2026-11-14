"""Chapter 15 lab: Listings 15.9, 15.10, 15.11, 15.12.

Generated from the chapter by tools/listings.py, so it matches the printed
code; regenerate it rather than editing it. Run it from this folder or from
the Labs folder, for example: python labs/ch15_serving_lab.py
"""
import os
import sys

# Make tinyml, in the folder above this one, importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ---- Listing 15.9 ----
import os
import tempfile

from tinyml.base import Pipeline
from tinyml.data import make_churn
from tinyml.linear import LogisticRegression
from tinyml.metrics import accuracy, roc_auc
from tinyml.preprocessing import TableEncoder, train_test_split
from tinyml.serve import infer_schema, load_model, save_model

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
FEATURES = NUMERIC + CATEGORICAL

table, y = make_churn(n=5000, seed=0)
X_train, X_test, y_train, y_test = train_test_split(table, y, test_size=0.2,
                                                    seed=0, stratify=True)
X_train = {c: X_train[c] for c in FEATURES}  # drop the leaky column for good
model = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                  ("classify", LogisticRegression())]).fit(X_train, y_train)
p_test = model.predict_proba(X_test)[:, 1]
metrics = {"test_accuracy": round(accuracy(y_test, model.predict(X_test)), 4),
           "test_auc": round(roc_auc(y_test, p_test), 4)}

folder = tempfile.mkdtemp()
path = os.path.join(folder, "churn-model.json")
manifest = save_model(model, path, schema=infer_schema(X_train),
                      train_table=X_train, metrics=metrics,
                      notes="Churn risk for retention offers. Not for pricing.")
loaded, info = load_model(path, expected_sha256=manifest["model_sha256"])

print("saved", os.path.getsize(path), "bytes; sha256", info["model_sha256"][:16])
print("trained on", info["n_train"], "rows; data fingerprint",
      info["train_fingerprint"][:16])
print("metrics", info["metrics"])
same = (loaded.predict_proba(X_test) == model.predict_proba(X_test)).all()
print("loaded model gives identical probabilities:", bool(same))
print("schema for contract:", info["schema"]["contract"])


# ---- Listing 15.10 ----
import json
import urllib.error
import urllib.request

from tinyml.serve import PredictionApp, make_server, validate_input

customer = {"tenure_months": 3.0, "monthly_charges": 89.5, "total_charges": None,
            "contract": "month-to-month", "support_calls": 4.0,
            "payment_method": "electronic", "region": "south", "age": 34.0}
bad = dict(customer, contract="monthly", age="34", monthly_charges=900.0)
print(validate_input(customer, info["schema"]))
for problem in validate_input(bad, info["schema"])[0]:
    print("error:", problem)

app = PredictionApp(loaded, info)
server = make_server(app)  # a real HTTP server on a free local port
url = f"http://127.0.0.1:{server.server_address[1]}"


def post(path, payload):
    request = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


with urllib.request.urlopen(url + "/health", timeout=10) as response:
    print(response.status, json.load(response))
longer = dict(customer, contract="two-year")
status, reply = post("/predict", {"records": [customer, longer]})
print(status, reply["predictions"])
status, reply = post("/predict", {"records": [customer, bad]})
print(status, reply["error"], "in record", list(reply["details"]))
status, reply = post("/predict", {"rows": []})
print(status, reply["error"])
server.shutdown()
server.server_close()
print("records logged for monitoring:", len(app.log))


# ---- Listing 15.11 ----
from tinyml.monitor import DriftReport, psi

same_table, same_y = make_churn(n=5000, seed=1)              # a new sample, no drift
drift_table, drift_y = make_churn(n=5000, seed=2, drift=True)  # a year later

report = DriftReport(X_train)
print("New sample from the training distribution:")
print(report.compare(same_table, FEATURES))
print("\nDrifted sample:")
print(report.compare(drift_table, FEATURES))
print("\nColumns flagged:", report.drifted())

train_scores = model.predict_proba(X_train)[:, 1]
for name, t in [("same", same_table), ("drift", drift_table)]:
    scores = model.predict_proba(t)[:, 1]
    print(f"{name:>5}: score PSI {psi(train_scores, scores):.3f}, "
          f"mean score {scores.mean():.3f}")


# ---- Listing 15.12 ----
import numpy as np

from tinyml.base import clone
from tinyml.metrics import log_loss

# Simulated data lets us evaluate on large fresh samples, so that differences
# of a point or two aren't lost in the noise of a 1,000-row test set.
old_world, y_old = make_churn(n=20000, seed=5)
new_world, y_new = make_churn(n=20000, seed=6, drift=True)


class TrueModel:
    """The generative model in data.py: the best any model could do."""

    def __init__(self, drift):
        self.drift = drift

    def predict_proba(self, t):
        c, calls = t["contract"], t["support_calls"]
        z = (-1.2 + 1.5 * (c == "month-to-month") + 0.3 * (c == "one-year")
             - 0.045 * t["tenure_months"] + 0.03 * (t["monthly_charges"] - 60)
             + (0.60 if self.drift else 0.35) * calls
             + 0.8 * ((calls >= 4) & (c == "month-to-month"))
             + 0.5 * (t["payment_method"] == "electronic") - 0.01 * (t["age"] - 45))
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def predict(self, t):
        return (self.predict_proba(t)[:, 1] >= 0.5).astype(int)


def rows(table, n):
    return {c: table[c][:n] for c in FEATURES}


def report_row(name, fitted, table, y):
    p = fitted.predict_proba(table)[:, 1]
    print(f"{name:<30}{accuracy(y, fitted.predict(table)):>6.3f}{roc_auc(y, p):>7.3f}"
          f"{log_loss(y, p):>7.3f}{p.mean():>7.3f}{y.mean():>7.3f}")


print(f"{'model':<30}{'acc':>6}{'AUC':>7}{'loss':>7}{'mean p':>7}{'rate':>7}")
report_row("true model, old world", TrueModel(False), old_world, y_old)
report_row("old model, old world", model, old_world, y_old)
report_row("true model, new world", TrueModel(True), new_world, y_new)
report_row("old model, new world", model, new_world, y_new)
for n in (500, 4000):  # labels arrive over time after the drift
    retrained = clone(model).fit(rows(drift_table, n), drift_y[:n])
    report_row(f"retrained on {n} new rows", retrained, new_world, y_new)
both = {c: np.concatenate([X_train[c], drift_table[c][:4000]]) for c in FEATURES}
combined = clone(model).fit(both, np.concatenate([y_train, drift_y[:4000]]))
report_row("old + 4000 new rows", combined, new_world, y_new)
