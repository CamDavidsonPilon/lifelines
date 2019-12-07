# -*- coding: utf-8 -*-
import inspect
import pandas as pd

try:
    from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin
except ImportError:
    raise ImportError("scikit-learn must be installed on the local system to use this utility class.")
from . import concordance_index

__all__ = ["sklearn_adapter"]


def filter_kwargs(f, kwargs):
    s = inspect.signature(f)
    res = {k: kwargs[k] for k in s.parameters if k in kwargs}
    return res


class _SklearnModel(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    def __init__(self, **kwargs):
        self._params = kwargs
        self.lifelines_model = self.lifelines_model(**filter_kwargs(self.lifelines_model.__init__, self._params))
        self._params["duration_col"] = "duration_col"
        self._params["event_col"] = self._event_col

    @property
    def _yColumn(self):
        return self._params["duration_col"]

    @property
    def _eventColumn(self):
        return self._params["event_col"]

    def fit(self, X, y=None):
        """

        Parameters
        -----------

        X: DataFrame
            must be a pandas DataFrame (with event_col included, if applicable)

        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame. Got type: {}".format(type(X)))

        X = X.copy()

        if y is not None:
            X.insert(len(X.columns), self._yColumn, y, allow_duplicates=False)

        fit = getattr(self.lifelines_model, self._fit_method)
        self.lifelines_model = fit(df=X, **filter_kwargs(fit, self._params))
        return self

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self.lifelines_model, key, value)
        return self

    def get_params(self, deep=True):
        out = {}
        for name, p in inspect.signature(self.lifelines_model.__init__).parameters.items():
            if p.kind < 4:  # ignore kwargs
                out[name] = getattr(self.lifelines_model, name)
        return out

    def predict(self, X, **kwargs):
        """
        Parameters
        ------------
        X: DataFrame or numpy array

        """
        predictions = getattr(self.lifelines_model, self._predict_method)(X, **kwargs).squeeze().values
        return predictions

    def score(self, X, y, **kwargs):
        """

        Parameters
        -----------

        X: DataFrame
            must be a pandas DataFrame (with event_col included, if applicable)

        """
        rest_columns = list(set(X.columns) - {self._yColumn, self._eventColumn})

        x = X.loc[:, rest_columns]
        e = X.loc[:, self._eventColumn] if self._eventColumn else None

        if y is None:
            y = X.loc[:, self._yColumn]

        if callable(self._scoring_method):
            res = self._scoring_method(y, self.predict(x, **kwargs), event_observed=e)
        else:
            raise ValueError()
        return res


def sklearn_adapter(fitter, event_col=None, predict_method="predict_expectation", scoring_method=concordance_index):
    """
    This function wraps lifelines models into a scikit-learn compatible API. The function returns a
    class that can be instantiated with parameters (similar to a scikit-learn class).

    Parameters
    ----------

    fitter: class
        The class (not an instance) to be wrapper. Example: ``CoxPHFitter`` or ``WeibullAFTFitter``
    event_col: string
        The column in your DataFrame that represents (if applicable) the event column
    predict_method: string
        Can be the string ``"predict_median", "predict_expectation"``
    scoring_method: function
        Provide a way to produce a ``score`` on the scikit-learn model. Signature should look like (durations, predictions, event_observed=None)

    """
    name = "SkLearn" + fitter.__name__
    klass = type(
        name,
        (_SklearnModel,),
        {
            "lifelines_model": fitter,
            "_event_col": event_col,
            "_predict_method": predict_method,
            "_fit_method": "fit",
            "_scoring_method": staticmethod(scoring_method),
        },
    )
    globals()[klass.__name__] = klass
    return klass
