"""Linear models: regression (Chapter 4) and regularized and classification models (Chapter 5)."""
import numpy as np

from tinyml.base import Estimator
from tinyml.optimize import gradient_descent


def add_intercept(X):
    """Prepends a column of ones, so that w[0] acts as the intercept."""
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    return np.column_stack([np.ones(len(X)), X])


def mse_objective(A, y):
    """Returns objective(w) -> (loss, gradient) for the mean squared error
    of the predictions A @ w, for use with the optimizers in tinyml.optimize.
    """
    n = len(y)

    def objective(w):
        residual = A @ w - y
        return residual @ residual / n, (2.0 / n) * (A.T @ residual)

    return objective


class LinearRegression(Estimator):
    """Ordinary least squares, fitted in closed form or by gradient descent.

    method="normal" solves the least-squares problem directly with
    np.linalg.lstsq, which is accurate even when columns are nearly
    collinear. method="gd" runs n_iter steps of gradient descent with step
    size lr on the mean squared error; scale the features first.
    """

    def __init__(self, method="normal", lr=0.1, n_iter=1000):
        self.method = method
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        A, y = add_intercept(X), np.asarray(y, dtype=float)
        if self.method == "normal":
            w = np.linalg.lstsq(A, y, rcond=None)[0]
        elif self.method == "gd":
            w, self.loss_history_ = gradient_descent(
                mse_objective(A, y), np.zeros(A.shape[1]),
                lr=self.lr, n_iter=self.n_iter)
        else:
            raise ValueError(f"unknown method '{self.method}'")
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return add_intercept(X) @ np.r_[self.intercept_, self.coef_]


import itertools

import numpy as np

from tinyml.base import Estimator, Transformer


def _as_matrix(X):
    """Returns X as a 2-D float array; a 1-D input becomes one column."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


class PolynomialFeatures(Transformer):
    """Expands the columns into all products of them up to a given degree.

    For columns a and b with degree=2, the output columns are a, b, a*a, a*b,
    and b*b. There is no constant column, because each model in tinyml fits
    its own intercept.
    """

    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        n_features = _as_matrix(X).shape[1]
        self.combinations_ = [
            combo
            for d in range(1, self.degree + 1)
            for combo in itertools.combinations_with_replacement(range(n_features), d)]
        return self

    def transform(self, X):
        X = _as_matrix(X)
        return np.column_stack([X[:, list(combo)].prod(axis=1)
                                for combo in self.combinations_])

    def feature_names(self, input_names):
        """Readable names for the output columns, such as 'age*tenure'."""
        return ["*".join(input_names[i] for i in combo)
                for combo in self.combinations_]


class Ridge(Estimator):
    """Least squares with an L2 penalty, solved in closed form.

    Minimizes ||y - Xw - b||^2 + alpha * ||w||^2. The intercept b is not
    penalized: centering X and y removes it from the problem, and it is
    recovered afterward from the means.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = _as_matrix(X), np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        # lstsq rather than solve, so alpha=0 still works when columns are
        # collinear (it then returns the minimum-norm solution).
        self.coef_ = np.linalg.lstsq(A, Xc.T @ yc, rcond=None)[0]
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return _as_matrix(X) @ self.coef_ + self.intercept_


def soft_threshold(z, t):
    """Shrinks z toward zero by t, and sets it to zero if |z| <= t."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


class Lasso(Estimator):
    """Least squares with an L1 penalty, fitted by cyclic coordinate descent.

    Minimizes (1 / 2n) ||y - Xw - b||^2 + alpha * ||w||_1, the scaling used
    by scikit-learn. Each pass updates one weight at a time with the exact
    one-dimensional minimizer, which is a soft-thresholding step.
    """

    def __init__(self, alpha=1.0, n_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = _as_matrix(X), np.asarray(y, dtype=float)
        n, d = X.shape
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc = X - x_mean
        col_sq = (Xc ** 2).sum(axis=0) / n
        w = np.zeros(d)
        residual = y - y_mean  # y - Xw - b, kept up to date as w changes
        for sweep in range(self.n_iter):
            largest_change = 0.0
            for j in range(d):
                if col_sq[j] == 0:
                    continue  # a constant column carries no information
                old = w[j]
                rho = Xc[:, j] @ residual / n + col_sq[j] * old
                w[j] = soft_threshold(rho, self.alpha) / col_sq[j]
                if w[j] != old:
                    residual -= Xc[:, j] * (w[j] - old)
                    largest_change = max(largest_change, abs(w[j] - old))
            if largest_change < self.tol:
                break
        self.n_iter_ = sweep + 1
        self.coef_ = w
        self.intercept_ = y_mean - x_mean @ w
        return self

    def predict(self, X):
        return _as_matrix(X) @ self.coef_ + self.intercept_


def sigmoid(z):
    """The logistic function 1 / (1 + e^-z), computed without overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


class LogisticRegression(Estimator):
    """Binary logistic regression with an L2 penalty.

    Minimizes the mean log loss plus (alpha / 2) * ||w||^2; the intercept is
    not penalized. method="newton" uses Newton's method (also known as
    iteratively reweighted least squares) and usually converges in under 10
    iterations; method="gd" uses gradient descent with step size lr, and
    needs scaled features and many more iterations. If C is given, it
    overrides alpha with alpha = 1 / (C * n), scikit-learn's convention.
    """

    def __init__(self, alpha=1e-3, C=None, method="newton", lr=0.5,
                 n_iter=100, tol=1e-8):
        self.alpha = alpha
        self.C = C
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = _as_matrix(X), np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression is binary; use SoftmaxRegression")
        t = (y == self.classes_[1]).astype(float)
        n, d = X.shape
        alpha = self.alpha if self.C is None else 1.0 / (self.C * n)
        A = np.column_stack([np.ones(n), X])  # column 0 is the intercept
        penalty = alpha * np.r_[0.0, np.ones(d)]

        def objective(theta):  # log(1 + e^z) - t*z is the log loss
            z = A @ theta
            return np.mean(np.logaddexp(0.0, z) - t * z) + 0.5 * penalty @ theta ** 2

        theta = np.zeros(d + 1)
        for iteration in range(self.n_iter):
            p = sigmoid(A @ theta)
            gradient = A.T @ (p - t) / n + penalty * theta
            if self.method == "newton":
                hessian = (A.T * (p * (1 - p))) @ A / n + np.diag(penalty)
                hessian += 1e-10 * np.eye(d + 1)  # keeps it invertible
                step = np.linalg.solve(hessian, gradient)
                # A full Newton step can overshoot when the classes are nearly
                # separable, so halve it until the objective decreases.
                current = objective(theta)
                while objective(theta - step) > current and np.abs(step).max() > 1e-12:
                    step = step / 2
            else:
                step = self.lr * gradient
            theta -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.n_iter_ = iteration + 1
        self.intercept_, self.coef_ = theta[0], theta[1:]
        return self

    def decision_function(self, X):
        """The log-odds of class classes_[1]."""
        return _as_matrix(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return self.classes_[(self.predict_proba(X)[:, 1] >= threshold).astype(int)]


def softmax(Z):
    """Row-wise softmax. Subtracting each row's maximum avoids overflow."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


class SoftmaxRegression(Estimator):
    """Multiclass logistic regression, fitted by gradient descent.

    Each class k has a weight vector and an intercept, and P(y = k | x) is
    the softmax of the k scores. Minimizes the mean cross-entropy plus
    (alpha / 2) * ||W||^2. Scale the features first, as with any
    gradient-descent model.
    """

    def __init__(self, alpha=1e-3, lr=0.5, n_iter=2000, tol=1e-6):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = _as_matrix(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        T = (y[:, None] == self.classes_[None, :]).astype(float)  # one-hot
        W = np.zeros((d, len(self.classes_)))
        b = np.zeros(len(self.classes_))
        for iteration in range(self.n_iter):
            G = (softmax(X @ W + b) - T) / n  # gradient of the loss wrt scores
            grad_W = X.T @ G + self.alpha * W
            grad_b = G.sum(axis=0)
            W -= self.lr * grad_W
            b -= self.lr * grad_b
            if max(np.abs(grad_W).max(), np.abs(grad_b).max()) < self.tol:
                break
        self.n_iter_ = iteration + 1
        self.coef_, self.intercept_ = W, b
        return self

    def predict_proba(self, X):
        return softmax(_as_matrix(X) @ self.coef_ + self.intercept_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
