# -*- coding: utf-8 -*-
import inspect
import pandas as pd

try:
    from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin
except ImportError:
    raise ImportError("scikit-learn must be installed on the local system to use this utility class.")
from . import concordance_index
from ..fitters import BaseFitter

__all__ = ("sklearn_adapter", "LifelinesSKLearnAdapter")


def filter_kwargs(f, kwargs):
    s = inspect.getfullargspec(f)
    if s.varkw:
        return kwargs
    else:
        allArgs = set()
        if s.args:
            allArgs |= set(s.args)
        if s.kwonlyargs:
            allArgs |= set(s.kwonlyargs)
        fedArgs = set(kwargs)
        redundantArgs = fedArgs - allArgs
        presentArgs = fedArgs & allArgs
        #if redundantArgs:
        #    warnings.warn("Following args are redundant for " + str(f) + str(inspect.signature(f)) + ": " + repr(redundantArgs))
        res = {k: kwargs[k] for k in presentArgs}
        return res


class LifelinesSKLearnAdapter(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    def __init__(self, lifelines_model=None, call_params=None, y_arg_names="duration_col", event_arg_name="event_col", remap=None, concordance_remap=None, predict_method="predict_expectation", fit_method="fit", etalon_func=None, **kwargs):
        """
        Parameters
        ----------

        lifelines_model: BaseFitter
            lifelines model to be wrapper. Example: ``CoxPHFitter(...)`` or ``WeibullAFTFitter(...)``
        params: 
            args to pass to the functions like `fit`
        event_col: string
            The column in your DataFrame that represents (if applicable) the event column
        predict_method: `str` or `callable`
            Can be a name of method like ``"predict_median", "predict_expectation"`` or a callable
        y_arg_names: string or a tuple of strings
            name(s) of columns considered to be targets. It can be `duration_col` or fields for parametric lifelines_models params or something else. Heavily depends on a lifelines_model.
        event_arg_name: string or None
            column name to pass censorship status to a lifelines_model
        remap: dict or None
            sometimes you need to rename params.
        concordance_remap: callable
            a function to remap concordance. For example, if you compare survival times to hazards you need `lambda sself, c: 1. - c`
        etalon_func: callable
            Can be used when `y_arg_names` are not survival times but parameters of a parametric lifelines_model.
        _sklearnParamsWorkaround: dict
            don't touch it - it is for sklearn, not for you
        """
        self.y_arg_names = y_arg_names
        self.event_arg_name = event_arg_name
        if remap is None:
            remap = {}

        self.remap = remap

        if etalon_func is None:

            def etalon_func(sself, Y):
                return Y

        self.etalon_func = etalon_func

        if concordance_remap is None:

            def concordance_remap(sself, c):
                return c

        self.concordance_remap = concordance_remap
        self.call_params = call_params
        
        ##############################################################

        if callable(lifelines_model):
            if isinstance(lifelines_model, type):
                self._init_params = filter_kwargs(lifelines_model.__init__, kwargs)
            else:
                self._init_params = filter_kwargs(lifelines_model, kwargs)
            
            self.lifelines_model = lifelines_model(**self._init_params)
        elif isinstance(lifelines_model, BaseFitter):
            self.lifelines_model = lifelines_model
        else:
            raise ValueError("`lifelines_model` must be either a callable or an object of class `BaseFitter`")

        #self.call_params["duration_col"] = "duration_col"
        #self.call_params[event_arg_name] = self._event_col

        if isinstance(y_arg_names, str):
            assert y_arg_names in self.call_params
        elif isinstance(y_arg_names, list):
            for yArgName in y_arg_names:
                assert yArgName in self.call_params

        ###########################
        self._predict_method_name = None
        if isinstance(predict_method, str):
            self._predict_method_name = predict_method
            predict_method_bound = getattr(self.lifelines_model, self._predict_method_name)

            def predict_method(sself, X):
                #return predict_method_bound(X)[0].values # shit, we cannot use that
                #predict_method_bound=getattr(sself.lifelines_model, sself._predict_method_name)
                return predict_method_bound(X)[0].values
        self.predict_method = predict_method
        
        self._fit_method_name = None
        if isinstance(fit_method, str):
            self._fit_method_name = fit_method
            fit_method_bound = getattr(self.lifelines_model, self._fit_method_name)

            def fit_method(sself, X, call_params):
                #return predict_method_bound(X)[0].values # shit, we cannot use that
                #fit_method_bound = getattr(sself.lifelines_model, sself._fit_method_name)
                return fit_method_bound(X, **filter_kwargs(fit_method_bound, call_params))

        self.fit_method = fit_method

    _essential_params = tuple(set(inspect.signature(__init__).parameters) - {"self", "kwargs"})

    @property
    def _event_col(self):
        return self.call_params[self.event_arg_name] if self.event_arg_name else None

    @property
    def _y_columns(self):
        if isinstance(self.y_arg_names, str):
            return self.call_params[self.y_arg_names]
        elif isinstance(self.y_arg_names, list):
            return [self.call_params[yArgName] for yArgName in self.y_arg_names]

    def Xye_split(self, pds):
        result_columns = {self._event_col}
        if isinstance(self._y_columns, str):
            result_columns |= {self._y_columns}
        elif isinstance(self._y_columns, list):
            result_columns |= set(self._y_columns)
        
        rest_columns = list(set(pds.columns) - result_columns)
        x = pds.loc[:, rest_columns]
        y = self.etalon_func(self, pds.loc[:, self._y_columns])
        e = pds.loc[:, self._event_col] if self._event_col else None
        return x, y, e

    def fit(self, X, y=None, sample_weight=None):
        """

        Parameters
        -----------

        X: DataFrame
            must be a pandas DataFrame (with event_col included, if applicable)

        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")

        X = X.copy()

        if y is not None:
            X.insert(len(X.columns), self._y_columns, y, allow_duplicates=False)

        params = type(self.call_params)(self.call_params)

        for nm, newNm in self.remap.items():
            params[newNm] = params[nm]
            del params[nm]
        
        self.lifelines_model = self.fit_method(self, X, self.call_params)

    def set_params(self, **params):
        for key, value in inspect.signature(self.lifelines_model.__init__).parameters:
            setattr(self.lifelines_model, key, value)
        set_essential_params(params)
        return self
    
    def get_essential_params(self):
        return {k:getattr(self, k) for k in self.__class__._essential_params}

    def set_essential_params(self, params):
        for k in self.__class__._essential_params:
            setattr(self, k, params[k])

    def get_params(self, deep=True):
        out = {}
        for k in inspect.signature(self.lifelines_model.__init__).parameters:
            out[k] = getattr(self.lifelines_model, k)
        out.update(self.get_essential_params())
        return out

    def predict(self, X):
        """
        Parameters
        ------------
        X: DataFrame or numpy array

        """
        return self.predict_method(self, X)

    def score(self, X, y=None, sample_weight=None):
        """

        Parameters
        -----------

        X: DataFrame
            must be a pandas DataFrame (with event_col included, if applicable)

        """
        if y is None:
            X, etalon, event_column = self.Xye_split(X)
        else:
            etalon = y

        predicted = self.predict(X)
        res = concordance_index(etalon, predicted, event_observed=event_column)
        res = self.concordance_remap(self, res)
        return res

def sklearn_adapter(fitter, event_col=None, predict_method="predict_expectation"):
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

    """
    name = "SkLearn" + fitter.__name__
    klass = type(
        name,
        (LifelinesSKLearnAdapter,),
        {"lifelines_model": fitter, "_event_col": event_col, "_predict_method_name": predict_method, "_fit_method_name": "fit"},
    )
    globals()[klass.__name__] = klass
    return klass
