# -*- coding: utf-8 -*-
import inspect
import pandas

try:
    from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin
except ImportError:
    raise ImportError("scikit-learn must be installed on the local system to use this utility class.")
from . import concordance_index

__all__ = ["sklearn_adapter"]


def filter_kwargs(f, kwargs):
    s = inspect.signature(f)
    res = {k: kwargs[k] for k in s.parameters if k in kwargs}
    return res


def sklearn_adapter(fitter, duration_col, event_col, predictor="predict_expectation"):
    class _SklearnModel(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
        def __init__(self, **kwargs):
            self._params = kwargs
            self._fitter = fitter(**filter_kwargs(fitter.__init__, self._params))

            self._params["duration_col"] = duration_col
            self._params["event_col"] = event_col

            self._predict_function_name = predictor

        @property
        def _yColumn(self):
            return self._params["duration_col"]

        @property
        def _eventColumn(self):
            return self._params["event_col"]

        def fit(self, X, y=None, sample_weight=None):
            X = X.copy()

            if y is not None:
                X.insert(len(X.columns), self._yColumn, y, allow_duplicates=False)

            self._fitter.fit(df=X, **filter_kwargs(self._fitter.fit, self._params))
            return self

        def get_params(self, deep=True):
            out = {}
            for k in inspect.signature(self._fitter.__init__).parameters:
                out[k] = getattr(self._fitter, k)
            return out

        def predict(self, X):
            """lifelines-expected function to predict expectation"""
            return getattr(self._fitter, self._predict_function_name)(X)[0].values

        def score(self, X, y, sample_weight=None):
            rest_columns = list(set(X.columns) - {self._yColumn, self._eventColumn})

            x = X.loc[:, rest_columns]
            e = X.loc[:, self._eventColumn] if self._eventColumn else None

            if y is None:
                y = X.loc[:, self._yColumn]

            res = concordance_index(y, self.predict(x), event_observed=e)
            return res

    _SklearnModel.__name__ = "%s" % fitter.__name__
    return _SklearnModel
