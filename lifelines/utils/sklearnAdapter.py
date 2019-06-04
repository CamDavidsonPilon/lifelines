import inspect
import pandas
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, MultiOutputMixin
from . import concordance_index


def filterKwArgs(f, kwargs):
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
        #    warnings.warn("Following args are redundant for "+str(f)+str(inspect.signature(f))+": "+repr(redundantArgs))
        res = {k: kwargs[k] for k in presentArgs}
        return res


class LifelinesSKLearnAdapter(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    __slots__ = ("fitter", "params", "yArgName", "eventArgName", "remap", "concordanceRemap")
    def __init__(self, fitter, params, yArgName="duration_col", eventArgName="event_col", remap=None, concordanceRemap=None):
        self.fitter = fitter
        assert yArgName in params
        self.params = params
        self.yArgName = yArgName
        self.eventArgName = eventArgName
        if remap is None:
            remap = {}
        self.remap = remap
        if concordanceRemap is None:
            concordanceRemap = lambda sself, c: c
        self.concordanceRemap = concordanceRemap

    @property
    def yColumn(self):
        return self.params[self.yArgName]

    @property
    def eventColumn(self):
        return self.params[self.eventArgName] if self.eventArgName else None

    def fit(self, X, y=None, sample_weight=None):
        """y has the default value to allow a keyword one because sklearn pass the stuff as to a keyword one"""
        if y is not None:
            X.insert(len(X.columns), self.yColumn, y, allow_duplicates=False)

        params = type(self.params)(self.params)

        for nm, newNm in self.remap.items():
            params[newNm] = params[nm]
            del params[nm]

        #print(X)
        self.fitter.fit(df=X, **filterKwArgs(self.fitter.fit, self.params))
        return self

    def predict(self, X):
        """lifelines-expected function to predict expectation"""
        #print(X)
        return self.fitter.predict_expectation(X)[0]

    def xyESplit(self, pds):
        restColumns = list(set(pds.columns) - {self.yColumn, self.eventColumn})
        x = pds.loc[:, restColumns]
        y = pds.loc[:, self.yColumn]
        e = pds.loc[:, self.eventColumn] if self.eventColumn else None
        return x, y, e

    def score(self, X, y=None, sample_weight=None):
        if y is None:
            X, etalonExpectation, eventColumn = self.xyESplit(X)
        else:
            etalonExpectation = y
        predictedExpectation = self.predict(X)
        res = concordance_index(etalonExpectation, predictedExpectation, event_observed=eventColumn)
        res = self.concordanceRemap(self, res)
        return res
