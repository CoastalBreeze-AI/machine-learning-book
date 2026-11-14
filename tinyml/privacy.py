"""Differential privacy with the Laplace mechanism (Chapter 16).

A mechanism is epsilon-differentially private if changing one person's
record changes the probability of any output by at most a factor e^epsilon.
Smaller epsilon means more privacy and, as the lab shows, more noise.
"""
import numpy as np


def laplace_mechanism(value, sensitivity, epsilon, rng):
    """Releases value + Laplace noise with scale sensitivity / epsilon.

    sensitivity is the most the true value can change when one person's
    record changes. The guarantee holds only if it is a true upper bound.
    """
    if epsilon <= 0 or sensitivity < 0:
        raise ValueError("epsilon must be positive and sensitivity non-negative")
    return value + rng.laplace(0.0, sensitivity / epsilon)


def dp_mean(x, lower, upper, epsilon, rng):
    """An epsilon-DP estimate of the mean of x.

    Each value is clipped to [lower, upper], which bounds one person's
    influence on the mean to (upper - lower) / n. The bounds must be chosen
    without looking at the data, or the choice itself leaks information.
    This version treats n as public (neighboring datasets differ by changing
    one record, not by adding or removing one).
    """
    x = np.clip(np.asarray(x, dtype=float), lower, upper)
    sensitivity = (upper - lower) / len(x)
    return float(laplace_mechanism(x.mean(), sensitivity, epsilon, rng))


def dp_count(mask, epsilon, rng):
    """An epsilon-DP count. Adding or removing a person changes it by 1."""
    return float(laplace_mechanism(float(np.sum(mask)), 1.0, epsilon, rng))
