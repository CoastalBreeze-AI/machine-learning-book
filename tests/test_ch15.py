"""Tests for Chapter 15: tinyml/serve.py and tinyml/monitor.py.

Run from Instructor Materials/Labs:  python -m unittest tests.test_ch15
"""
import json
import math
import os
import tempfile
import unittest
import urllib.request

import numpy as np

from tinyml.base import Estimator, Pipeline
from tinyml.data import make_churn
from tinyml.monitor import DriftReport, categorical_psi, chi_square, ks_statistic, psi
from tinyml.preprocessing import TableEncoder
from tinyml.serve import (PredictionApp, fingerprint, from_jsonable, infer_schema,
                          load_model, make_server, records_to_table, save_model,
                          to_jsonable, validate_input)

try:
    from tinyml.linear import LogisticRegression
except ImportError:  # Chapter 5 not assembled yet
    LogisticRegression = None

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "age"]
CATEGORICAL = ["contract", "payment_method", "region"]
FEATURES = NUMERIC + CATEGORICAL


class Untrusted(Estimator):
    def __init__(self, k=1):
        self.k = k


def churn(n=2000, seed=0, drift=False):
    table, y = make_churn(n=n, seed=seed, drift=drift)
    return {c: table[c] for c in FEATURES}, y


@unittest.skipIf(LogisticRegression is None, "tinyml.linear is not available")
class TestSaveLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.y = churn()
        cls.model = Pipeline([("encode", TableEncoder(NUMERIC, CATEGORICAL)),
                              ("classify", LogisticRegression())]).fit(cls.table, cls.y)
        cls.folder = tempfile.mkdtemp()
        cls.path = os.path.join(cls.folder, "m.json")
        cls.manifest = save_model(cls.model, cls.path, schema=infer_schema(cls.table),
                                  train_table=cls.table, metrics={"acc": 0.7})

    def test_round_trip_gives_identical_probabilities(self):
        loaded, manifest = load_model(self.path, self.manifest["model_sha256"])
        np.testing.assert_array_equal(loaded.predict_proba(self.table),
                                      self.model.predict_proba(self.table))
        self.assertEqual(manifest["n_train"], 2000)
        self.assertEqual(manifest["metrics"], {"acc": 0.7})

    def test_file_is_plain_json(self):
        with open(self.path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn("manifest", saved)
        self.assertIsInstance(saved["model"], str)

    def test_corrupted_file_is_rejected(self):
        with open(self.path, encoding="utf-8") as f:
            saved = json.load(f)
        saved["model"] = saved["model"].replace("0.", "1.", 1)
        bad = os.path.join(self.folder, "bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        with self.assertRaises(ValueError):
            load_model(bad)

    def test_expected_hash_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            load_model(self.path, expected_sha256="0" * 64)

    def test_app_predicts_and_validates(self):
        app = PredictionApp(self.model, self.manifest)
        record = {c: (v.item() if hasattr(v, "item") else v)
                  for c, v in ((c, self.table[c][0]) for c in FEATURES)}
        if isinstance(record["total_charges"], float) and math.isnan(record["total_charges"]):
            record["total_charges"] = None
        status, reply = app.handle("POST", "/predict",
                                   json.dumps({"records": [record, record]}).encode())
        self.assertEqual(status, 200)
        self.assertEqual(len(reply["predictions"]), 2)
        expected = self.model.predict_proba(records_to_table([record], self.manifest["schema"]))
        self.assertAlmostEqual(reply["predictions"][0]["churn_probability"],
                               round(float(expected[0, 1]), 4))
        self.assertEqual(len(app.log), 2)
        status, _ = app.handle("POST", "/predict",
                               json.dumps({"records": [dict(record, contract="x")]}).encode())
        self.assertEqual(status, 400)
        self.assertEqual(app.handle("POST", "/predict", b"not json")[0], 400)
        self.assertEqual(app.handle("POST", "/predict", b'{"records": []}')[0], 400)
        self.assertEqual(app.handle("GET", "/nowhere")[0], 404)
        self.assertEqual(app.handle("GET", "/health")[0], 200)

    def test_http_server_round_trip(self):
        server = make_server(PredictionApp(self.model, self.manifest))
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                self.assertEqual(json.load(response)["status"], "ok")
        finally:
            server.shutdown()
            server.server_close()


class TestSerialization(unittest.TestCase):
    def test_values_round_trip(self):
        value = {"a": np.array([[1.5, np.nan]]), "b": (1, "x"), "c": [np.array(["p", "q"])],
                 "d": None, "e": np.float64(2.5)}
        back = from_jsonable(json.loads(json.dumps(to_jsonable(value))))
        np.testing.assert_array_equal(back["a"], value["a"])
        self.assertEqual(back["b"], (1, "x"))
        self.assertEqual(back["c"][0].tolist(), ["p", "q"])
        self.assertEqual(back["e"], 2.5)

    def test_untrusted_class_is_refused(self):
        data = to_jsonable(Untrusted())
        with self.assertRaises(ValueError):
            from_jsonable(data)
        data["__estimator__"] = "os.system"
        with self.assertRaises(ValueError):
            from_jsonable(data)

    def test_non_learned_attribute_is_refused(self):
        data = to_jsonable(TableEncoder(["a"], []))
        data["state"]["__class__"] = 1
        with self.assertRaises(ValueError):
            from_jsonable(data)

    def test_fingerprint(self):
        a, _ = churn(100)
        b = {k: v.copy() for k, v in a.items()}
        self.assertEqual(fingerprint(a), fingerprint(b))
        b["age"][3] += 1
        self.assertNotEqual(fingerprint(a), fingerprint(b))


class TestValidation(unittest.TestCase):
    schema = {"x": {"type": "number", "min": 0.0, "max": 10.0, "nullable": False},
              "z": {"type": "number", "min": 0.0, "max": 10.0, "nullable": True},
              "c": {"type": "category", "values": ["a", "b"]}}

    def check(self, **record):
        return validate_input({"x": 1.0, "z": None, "c": "a", **record}, self.schema)

    def test_good_record(self):
        self.assertEqual(self.check(), ([], []))

    def test_errors(self):
        self.assertTrue(self.check(x=None)[0])
        self.assertTrue(self.check(x="1")[0])
        self.assertTrue(self.check(x=True)[0])
        self.assertTrue(self.check(x=float("nan"))[0])
        self.assertTrue(self.check(x=100.0)[0])
        self.assertTrue(self.check(c="d")[0])
        self.assertTrue(self.check(c=3)[0])
        self.assertTrue(validate_input({"x": 1.0, "c": "a"}, self.schema)[0])

    def test_warnings(self):
        errors, warnings = self.check(x=12.0, extra=1)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)

    def test_records_to_table(self):
        t = records_to_table([{"x": 1, "z": None, "c": "a"}], self.schema)
        self.assertTrue(np.isnan(t["z"][0]))
        self.assertEqual(t["c"][0], "a")


class TestDrift(unittest.TestCase):
    def test_psi_identical_is_zero_and_shift_is_large(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        self.assertAlmostEqual(psi(x, x), 0.0, places=10)
        self.assertGreater(psi(x, x + 1.0), 0.25)
        self.assertLess(psi(x, rng.normal(size=5000)), 0.02)

    def test_psi_counts_missing_values(self):
        x = np.arange(1000.0)
        y = x.copy()
        y[:300] = np.nan
        self.assertGreater(psi(x, y), 0.25)

    def test_categorical_psi_known_answer(self):
        ref = np.array(["a"] * 50 + ["b"] * 50)
        cur = np.array(["a"] * 75 + ["b"] * 25)
        expected = 0.25 * math.log(1.5) + 0.25 * math.log(2)
        self.assertAlmostEqual(categorical_psi(ref, cur), expected, places=10)

    def test_ks(self):
        d, p = ks_statistic([1, 2, 3], [4, 5, 6])
        self.assertEqual(d, 1.0)
        d, p = ks_statistic(np.arange(100.0), np.arange(100.0))
        self.assertEqual((d, p), (0.0, 1.0))
        rng = np.random.default_rng(1)
        d, p = ks_statistic(rng.normal(size=2000), rng.normal(0.3, 1, size=2000))
        self.assertLess(p, 1e-6)

    def test_chi_square_known_answer(self):
        ref = np.array(["a"] * 50 + ["b"] * 50)
        cur = np.array(["a"] * 75 + ["b"] * 25)
        stat, p = chi_square(ref, cur)
        self.assertAlmostEqual(stat, 40 / 3, places=10)
        exact = math.erfc(math.sqrt(stat / 2))  # chi-square with 1 dof
        self.assertLess(abs(math.log(p / exact)), 0.5)

    def test_drift_report(self):
        ref, _ = churn(5000, seed=0)
        same, _ = churn(5000, seed=1)
        drifted, _ = churn(5000, seed=2, drift=True)
        self.assertEqual(DriftReport(ref).compare(same).drifted(), [])
        report = DriftReport(ref).compare(drifted)
        self.assertIn("monthly_charges", report.drifted())
        self.assertEqual(len(str(report).splitlines()), len(FEATURES) + 1)


if __name__ == "__main__":
    unittest.main()
