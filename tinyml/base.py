"""The estimator interface shared by every model in tinyml (Chapter 3)."""
import copy
import inspect


class Estimator:
    """Base class for models. Subclasses implement fit() and predict().

    Constructor arguments are hyperparameters. They must be stored under the
    same names, unchanged, so that get_params() and clone() work. Anything
    learned from data is stored in an attribute whose name ends with an
    underscore, such as coef_, and is set only by fit().
    """

    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def get_params(self):
        # A class without its own __init__ has no hyperparameters. Skipping
        # *args and **kwargs handles object.__init__, which accepts both.
        params = inspect.signature(type(self).__init__).parameters.values()
        return {p.name: getattr(self, p.name) for p in params
                if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}

    def set_params(self, **params):
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"{type(self).__name__} has no parameter '{name}'")
            setattr(self, name, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class Transformer(Estimator):
    """Base class for preprocessing steps, which change X rather than predict y."""

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def clone(estimator):
    """Returns an unfitted copy with the same hyperparameters."""
    params = {k: copy.deepcopy(v) for k, v in estimator.get_params().items()}
    return type(estimator)(**params)


class Pipeline(Estimator):
    """Chains transformers and a final estimator, so they are fitted together.

    Fitting a pipeline fits each step only on the data passed to fit(), which
    is what keeps preprocessing from leaking information from test data.
    """

    def __init__(self, steps):
        self.steps = steps  # a list of (name, estimator) pairs

    def fit(self, X, y=None):
        self.fitted_steps_ = []
        for name, step in self.steps[:-1]:
            step = clone(step)
            X = step.fit_transform(X, y)
            self.fitted_steps_.append((name, step))
        name, final = self.steps[-1]
        final = clone(final).fit(X, y)
        self.fitted_steps_.append((name, final))
        return self

    def _transform_all_but_last(self, X):
        for _, step in self.fitted_steps_[:-1]:
            X = step.transform(X)
        return X

    def predict(self, X):
        return self.fitted_steps_[-1][1].predict(self._transform_all_but_last(X))

    def predict_proba(self, X):
        return self.fitted_steps_[-1][1].predict_proba(self._transform_all_but_last(X))

    def transform(self, X):
        return self.fitted_steps_[-1][1].transform(self._transform_all_but_last(X))

    def __getitem__(self, name):
        return dict(self.fitted_steps_)[name]
