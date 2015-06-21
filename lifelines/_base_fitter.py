# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from lifelines.plotting import plot_estimate
from lifelines.utils import _conditional_time_to_event_

class BaseFitter(object):

    def __init__(self, alpha=0.95):
        if not (0 < alpha <= 1.):
            raise ValueError('alpha parameter must be between 0 and 1.')
        self.alpha = alpha

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifelines.%s: fitted with %d observations, %d censored>""" % (
                classname, self.event_observed.shape[0], (1 - self.event_observed).sum())
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

class UnivariateFitter(BaseFitter):

    def _plot_estimate(self, *args):
        return plot_estimate(self, *args)

    def _subtract(self, estimate):
        class_name = self.__class__.__name__
        doc_string = """
            Subtract the %s of two %s objects.

            Parameters:
              other: an %s fitted instance.

            """ % (estimate, class_name, class_name)

        def subtract(other):
            self_estimate = getattr(self, estimate)
            other_estimate = getattr(other, estimate)
            new_index = np.concatenate((other_estimate.index, self_estimate.index))
            new_index = np.unique(new_index)
            return self_estimate.reindex(new_index, method='ffill') - \
                other_estimate.reindex(new_index, method='ffill')

        subtract.__doc__ = doc_string
        return subtract


    def _divide(self, estimate):
        class_name = self.__class__.__name__
        doc_string = """
            Divide the %s of two %s objects.

            Parameters:
              other: an %s fitted instance.

            """ % (estimate, class_name, class_name)

        def divide(other):
            self_estimate = getattr(self, estimate)
            other_estimate = getattr(other, estimate)
            new_index = np.concatenate((other_estimate.index, self_estimate.index))
            new_index = np.unique(new_index)
            return self_estimate.reindex(new_index, method='ffill') / \
                other_estimate.reindex(new_index, method='ffill')

        divide.__doc__ = doc_string
        return divide


    def _predict(self, estimate, label):
        class_name = self.__class__.__name__
        doc_string = """
          Predict the %s at certain point in time.

          Parameters:
            time: a scalar or an array of times to predict the value of %s at.

          Returns:
            predictions: a scalar if time is a scalar, a numpy array if time in an array.
          """ % (class_name, class_name)

        def predict(time):
            predictor = lambda t: getattr(self, estimate).ix[:t].iloc[-1][label]
            try:
                return np.array([predictor(t) for t in time])
            except TypeError:
                return predictor(time)

        predict.__doc__ = doc_string
        return predict

    @property
    def conditional_time_to_event_(self):
        return _conditional_time_to_event_(self)