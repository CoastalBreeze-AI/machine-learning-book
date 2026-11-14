"""Saving, loading, validating, and serving models (Chapter 15).

Models are saved as JSON, not pickle. A JSON file can only describe data, so
loading one never runs code that came from the file. The price is that only
estimators from trusted modules can be rebuilt.
"""
import hashlib
import importlib
import json
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from tinyml.base import Estimator

TRUSTED_MODULES = ("tinyml.",)  # Only classes from these modules are rebuilt.


def to_jsonable(value):
    """Converts an estimator, array, or plain value to JSON-safe data."""
    if isinstance(value, Estimator):
        cls = type(value)
        state = {k: to_jsonable(v) for k, v in vars(value).items()
                 if k.endswith("_") and not k.startswith("_")}  # learned attributes
        return {"__estimator__": f"{cls.__module__}.{cls.__qualname__}",
                "params": to_jsonable(value.get_params()), "state": state}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": value.dtype.str,
                "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [to_jsonable(v) for v in value]}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {"__dict__": {str(k): to_jsonable(v) for k, v in value.items()}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"can't save a value of type {type(value).__name__}")


def _trusted_class(path):
    module_name, _, class_name = path.rpartition(".")
    if not module_name.startswith(TRUSTED_MODULES):
        raise ValueError(f"refusing to load class from untrusted module '{module_name}'")
    cls = getattr(importlib.import_module(module_name), class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Estimator)):
        raise ValueError(f"'{path}' is not an Estimator")
    return cls


def from_jsonable(data):
    """Inverts to_jsonable. Rebuilds only Estimator classes from trusted modules."""
    if isinstance(data, list):
        return [from_jsonable(v) for v in data]
    if not isinstance(data, dict):
        return data
    if "__estimator__" in data:
        cls = _trusted_class(data["__estimator__"])
        model = cls(**from_jsonable(data["params"]))
        for name, value in data["state"].items():
            if name.startswith("_") or not name.endswith("_"):
                raise ValueError(f"'{name}' is not a learned attribute")
            setattr(model, name, from_jsonable(value))
        return model
    if "__ndarray__" in data:
        array = np.array(data["__ndarray__"], dtype=np.dtype(data["dtype"]))
        return array.reshape(data["shape"])
    if "__tuple__" in data:
        return tuple(from_jsonable(v) for v in data["__tuple__"])
    if "__dict__" in data:
        return {k: from_jsonable(v) for k, v in data["__dict__"].items()}
    raise ValueError(f"unrecognized saved object with keys {sorted(data)}")


FORMAT = "tinyml-model/1"


def fingerprint(table):
    """A SHA-256 digest of a table's column names, types, and values.

    Two tables have the same fingerprint only if they hold the same data, so
    recording it lets you say exactly which data a model was trained on.
    """
    digest = hashlib.sha256()
    for name in sorted(table):
        column = np.asarray(table[name])
        digest.update(f"{name}:{column.dtype.str}:{len(column)};".encode())
        digest.update(column.tobytes())
    return digest.hexdigest()


def infer_schema(table):
    """Records each column's type and the values or range seen in training."""
    schema = {}
    for name, column in table.items():
        column = np.asarray(column)
        if column.dtype.kind in "fiu":
            values = column[~np.isnan(column.astype(float))]
            schema[name] = {"type": "number", "min": float(values.min()),
                            "max": float(values.max()),
                            "nullable": bool(len(values) < len(column))}
        else:
            schema[name] = {"type": "category",
                            "values": sorted(str(v) for v in np.unique(column))}
    return schema


def save_model(model, path, *, schema, train_table, metrics=None, notes=""):
    """Writes a fitted model and its manifest to one JSON file."""
    body = json.dumps(to_jsonable(model), sort_keys=True)
    manifest = {
        "format": FORMAT,
        "model_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "model_class": type(model).__name__,
        "schema": schema,
        "train_fingerprint": fingerprint(train_table),
        "n_train": len(next(iter(train_table.values()))),
        "versions": {"numpy": np.__version__, "python": platform.python_version()},
        "metrics": metrics or {},
        "notes": notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "model": body}, f, indent=1)
    return manifest


def load_model(path, expected_sha256=None):
    """Reads a model saved by save_model. Returns (model, manifest).

    The hash detects accidental corruption. It does not stop tampering by
    someone who can rewrite the whole file; for that, compare against a hash
    kept somewhere they can't reach (expected_sha256), or sign the file.
    """
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    manifest, body = saved["manifest"], saved["model"]
    if manifest.get("format") != FORMAT:
        raise ValueError(f"unknown format {manifest.get('format')!r}")
    actual = hashlib.sha256(body.encode()).hexdigest()
    if actual != manifest["model_sha256"]:
        raise ValueError("model hash doesn't match its manifest: file is corrupt")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("model hash doesn't match the expected hash")
    return from_jsonable(json.loads(body)), manifest


def validate_input(record, schema, range_slack=0.5):
    """Checks one record (a dict of plain values) against a schema.

    Returns (errors, warnings). Errors mean the record can't be scored
    safely. Warnings mean it can, but it lies outside what the model saw in
    training, which is worth counting even if each one is harmless.
    """
    errors, warnings = [], []
    for name, spec in schema.items():
        if name not in record:
            errors.append(f"missing column '{name}'")
            continue
        value = record[name]
        if spec["type"] == "number":
            if value is None:
                if not spec["nullable"]:
                    errors.append(f"'{name}' may not be null")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"'{name}' must be a number, got {type(value).__name__}")
            elif value != value or value in (float("inf"), float("-inf")):
                errors.append(f"'{name}' must be finite")
            else:
                width = spec["max"] - spec["min"]
                low = spec["min"] - range_slack * width
                high = spec["max"] + range_slack * width
                if not low <= value <= high:
                    errors.append(f"'{name}'={value} is far outside the training "
                                  f"range [{spec['min']:g}, {spec['max']:g}]")
                elif not spec["min"] <= value <= spec["max"]:
                    warnings.append(f"'{name}'={value} is outside the training range")
        elif not isinstance(value, str):
            errors.append(f"'{name}' must be a string")
        elif value not in spec["values"]:
            errors.append(f"'{name}'='{value}' is not a known category")
    for name in record:
        if name not in schema:
            warnings.append(f"ignoring unknown column '{name}'")
    return errors, warnings


def records_to_table(records, schema):
    """Converts a list of validated records to a table, with None as NaN."""
    table = {}
    for name, spec in schema.items():
        values = [record[name] for record in records]
        if spec["type"] == "number":
            table[name] = np.array([np.nan if v is None else v for v in values],
                                   dtype=float)
        else:
            table[name] = np.array(values, dtype=str)
    return table


class PredictionApp:
    """The service's logic, kept separate from HTTP so it's easy to test.

    handle() takes a method, a path, and a request body in bytes, and
    returns (status code, response dict). It also keeps a log of the inputs
    it scored, which is what a drift monitor reads later.
    """

    def __init__(self, model, manifest, threshold=0.5, max_records=1000):
        self.model, self.manifest = model, manifest
        self.threshold, self.max_records = threshold, max_records
        self.version = manifest["model_sha256"][:12]
        self.log = []  # one entry per scored record
        self.lock = threading.Lock()

    def handle(self, method, path, body=b""):
        if method == "GET" and path == "/health":
            return 200, {"status": "ok", "model_version": self.version}
        if method != "POST" or path != "/predict":
            return 404, {"error": f"no route for {method} {path}"}
        try:
            request = json.loads(body)
            records = request["records"]
            if not (isinstance(records, list) and 0 < len(records) <= self.max_records
                    and all(isinstance(r, dict) for r in records)):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return 400, {"error": "expected JSON {\"records\": [ {...}, ... ]} "
                                  f"with 1 to {self.max_records} records"}
        schema = self.manifest["schema"]
        problems = {}
        for i, record in enumerate(records):
            errors, _ = validate_input(record, schema)
            if errors:
                problems[i] = errors
        if problems:
            return 400, {"error": "invalid records", "details": problems}
        start = time.perf_counter()
        p = self.model.predict_proba(records_to_table(records, schema))[:, 1]
        elapsed_ms = 1000 * (time.perf_counter() - start)
        with self.lock:
            self.log.extend({"record": r, "score": float(s)} for r, s in zip(records, p))
        return 200, {"model_version": self.version, "latency_ms": round(elapsed_ms, 3),
                     "predictions": [{"churn_probability": round(float(s), 4),
                                      "churn": int(s >= self.threshold)} for s in p]}


def make_server(app, host="127.0.0.1", port=0):
    """Wraps an app in a threaded HTTP server. Port 0 picks a free port."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply(*app.handle("GET", self.path))

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= 10_000_000:
                return self._reply(413, {"error": "bad or oversized request body"})
            self._reply(*app.handle("POST", self.path, self.rfile.read(length)))

        def log_message(self, *args):  # keep the console quiet
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
