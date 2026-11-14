"""Synthetic datasets for the labs. Chapter 3 introduces make_churn.

Because the data is simulated, the true causes of churn are known. That lets
you check whether a model, an explanation, or a drift detector found what is
really there, which is impossible with most real datasets.
"""
import numpy as np

CONTRACTS = np.array(["month-to-month", "one-year", "two-year"])
PAYMENTS = np.array(["electronic", "mailed_check", "bank_transfer", "credit_card"])
REGIONS = np.array(["north", "south", "east", "west"])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_churn(n=5000, seed=0, drift=False):
    """Simulates n customers of a subscription service.

    Returns (table, y). The table is a dict that maps column names to NumPy
    arrays of length n. Numeric columns use NaN for missing values, and
    categorical columns hold strings. y is 1 if the customer churned.

    With drift=True, prices rise, the customer mix changes, and support calls
    become a stronger cause of churn, as might happen a year later.
    """
    rng = np.random.default_rng(seed)

    region = rng.choice(REGIONS, size=n, p=[0.30, 0.25, 0.25, 0.20])
    age = np.clip(np.round(rng.normal(45, 15, n)), 18, 90)

    # Older customers are somewhat more likely to sign long contracts.
    long_bias = 0.6 * (age - 45) / 15
    m2m_share = 0.62 if drift else 0.52
    logits = np.column_stack([np.full(n, np.log(m2m_share / (1 - m2m_share)) + 0.6),
                              0.3 + 0.5 * long_bias, 0.0 + long_bias])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    contract_index = (rng.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    contract = CONTRACTS[contract_index]

    mean_tenure = np.array([14.0, 30.0, 45.0])[contract_index]
    tenure = np.clip(np.round(rng.exponential(mean_tenure)), 1, 72)

    pay_p = [0.45, 0.20, 0.18, 0.17] if drift else [0.35, 0.25, 0.20, 0.20]
    payment = rng.choice(PAYMENTS, size=n, p=pay_p)

    monthly = 20 + 80 * rng.beta(2.0, 2.5, n) + (8.0 if drift else 0.0)
    monthly = np.round(monthly, 2)
    total = np.round(monthly * tenure * rng.normal(1.0, 0.03, n), 2)

    # New customers often have no billing history yet, so total_charges is
    # missing more often when tenure is short: missing at random, given tenure.
    p_missing = np.where(tenure <= 2, 0.35, 0.02)
    total = np.where(rng.random(n) < p_missing, np.nan, total)

    senior = age >= 65
    call_rate = 0.8 + 0.015 * (monthly - 60) + 0.6 * senior + 0.2 * (region == "south")
    support_calls = rng.poisson(np.clip(call_rate, 0.05, None)).astype(float)

    # The true model of churn. Nothing outside this block affects the label.
    calls_weight = 0.60 if drift else 0.35
    z = (-1.2
         + 1.5 * (contract == "month-to-month")
         + 0.3 * (contract == "one-year")
         - 0.045 * tenure
         + 0.03 * (monthly - 60)
         + calls_weight * support_calls
         + 0.8 * ((support_calls >= 4) & (contract == "month-to-month"))
         + 0.5 * (payment == "electronic")
         - 0.01 * (age - 45))
    y = (rng.random(n) < _sigmoid(z)).astype(int)

    # A leaky column: retention offers are usually made after a customer asks
    # to cancel, so the column is recorded mostly for customers who churned.
    has_offer = rng.random(n) < np.where(y == 1, 0.85, 0.05)
    offer_days = np.where(has_offer, rng.integers(1, 31, n), np.nan)

    table = {
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract": contract,
        "support_calls": support_calls,
        "payment_method": payment,
        "region": region,
        "age": age,
        "retention_offer_days": offer_days,
    }
    return table, y


SHAPES = ("square", "circle", "triangle", "cross")


def make_shapes(n=2000, size=12, seed=0, noise=0.15):
    """Generates n small grayscale images of four shape classes (Chapter 12).

    Returns (X, y). X has shape (n, 1, size, size), with values near [0, 1];
    y holds class indices into SHAPES. Each image has a random position,
    size, and brightness, plus Gaussian pixel noise with standard deviation
    noise. The classes are balanced.
    """
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % len(SHAPES))
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    X = np.empty((n, 1, size, size))
    for i in range(n):
        r = rng.uniform(0.2, 0.36) * size                # half-width of the shape
        cy, cx = rng.uniform(r, size - 1 - r, size=2)    # keep it inside the frame
        dy, dx = rows - cy, cols - cx
        if y[i] == 0:      # square
            mask = np.maximum(np.abs(dx), np.abs(dy)) <= r
        elif y[i] == 1:    # circle
            mask = dx**2 + dy**2 <= r**2
        elif y[i] == 2:    # triangle, apex up
            mask = (np.abs(dy) <= r) & (np.abs(dx) <= (dy + r) / 2)
        else:              # cross, with arms a few pixels wide
            arm = max(0.6, 0.12 * size)
            mask = (((np.abs(dx) <= arm) & (np.abs(dy) <= r))
                    | ((np.abs(dy) <= arm) & (np.abs(dx) <= r)))
        brightness = rng.uniform(0.6, 1.0)
        X[i, 0] = brightness * mask + rng.normal(0.0, noise, (size, size))
    return X, y


def make_sort_task(n, length=6, vocab=10, seed=0):
    """A synthetic sequence task for Chapter 13: sort a list of digits.

    Returns (X, Y), integer arrays of shape (n, length). Each row of X holds
    tokens drawn uniformly from 0..vocab-1 (repeats allowed), and the
    matching row of Y is the same tokens in ascending order. np.flip(X, 1)
    gives a reversal task from the same inputs.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab, size=(n, length))
    return X, np.sort(X, axis=1)
