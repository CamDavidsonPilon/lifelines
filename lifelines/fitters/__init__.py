# -*- coding: utf-8 -*-
from __future__ import print_function
import collections

import numpy as np
import pandas as pd

from lifelines.plotting import plot_estimate
from lifelines.utils import qth_survival_times, _to_array


class BaseFitter(object):

    def __init__(self, alpha=0.95):
        if not (0 < alpha <= 1.):
            raise ValueError('alpha parameter must be between 0 and 1.')
        self.alpha = alpha

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifelines.%s: fitted with %d observations, %d censored>""" % (
                classname, self.event_observed.shape[0], self.event_observed.shape[0] - np.where(self.event_observed)[0].shape[0])
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s


class UnivariateFitter(BaseFitter):

    def _update_docstrings(self):
        # Update their docstrings
        self.__class__.subtract.__doc__ = self.subtract.__doc__.format(self._estimate_name,self.__class__.__name__)
        self.__class__.divide.__doc__ = self.divide.__doc__.format(self._estimate_name,self.__class__.__name__)
        self.__class__.predict.__doc__ = self.predict.__doc__.format(self.__class__.__name__)
        self.__class__.plot.__doc__ = plot_estimate.__doc__.format(self.__class__.__name__)

    def plot(self, *args, **kwargs):
        try:
            estimate = self._estimate_name
        except AttributeError:
            raise RuntimeError("Must call `fit` first!")
            
        return plot_estimate(self, *args, **kwargs)

    def subtract(self,other):
        """
        Subtract the {0} of two {1} objects.

            Parameters:
              other: an {1} fitted instance.
        """

        try:
            estimate = self._estimate_name
        except AttributeError:
            raise RuntimeError("Must call `fit` first!")
        
        self_estimate = getattr(self, self._estimate_name)
        other_estimate = getattr(other, other._estimate_name)
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        return pd.DataFrame(
            self_estimate.reindex(new_index, method='ffill').values -
            other_estimate.reindex(new_index, method='ffill').values,
            index=new_index,
            columns=['diff']
        )

    def divide(self, other):
        """
        Divide the {0} of two {1} objects.

        Parameters:
          other: an {1} fitted instance.

        """
        try:
            estimate = self._estimate_name
        except AttributeError:
            raise RuntimeError("Must call `fit` first!")
    
        self_estimate = getattr(self, self._estimate_name)
        other_estimate = getattr(other, other._estimate_name)
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        return pd.DataFrame(
            self_estimate.reindex(new_index, method='ffill').values /
            other_estimate.reindex(new_index, method='ffill').values,
            index=new_index,
            columns=['ratio']
        )

    def predict(self, times):
        """
        Predict the {0} at certain point in time. Uses a linear interpolation if
        points in time are not in the index.

        Parameters:
          time: a scalar or an array of times to predict the value of {0} at.

        Returns:
          predictions: a scalar if time is a scalar, a numpy array if time in an array.
        """ 
        try:
            estimate = self._estimate_name
        except AttributeError:
            raise RuntimeError("Must call `fit` first!")

        if callable(self._estimation_method):
            return pd.DataFrame(self._estimation_method(_to_array(times)), index=_to_array(times)).loc[times].squeeze()
        else:
            estimate = getattr(self, self._estimation_method)
            # non-linear interpolations can push the survival curves above 1 and below 0.
            return estimate.reindex(estimate.index.union(_to_array(times))).interpolate("index").loc[times].squeeze()
        
    @property
    def conditional_time_to_event_(self):
        return self._conditional_time_to_event_()

    def _conditional_time_to_event_(self):
        """
        Return a DataFrame, with index equal to survival_function_, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        individual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.

        Returns:
            conditional_time_to_: DataFrame, with index equal to survival_function_

        """
        age = self.survival_function_.index.values[:, None]
        columns = ['%s - Conditional time remaining to event' % self._label]
        return pd.DataFrame(qth_survival_times(self.survival_function_[self._label] * 0.5, self.survival_function_).sort_index(ascending=False).values,
                            index=self.survival_function_.index,
                            columns=columns) - age
