# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd

from lifelines._base_fitter import UnivariateFitter
from lifelines.utils import inv_normal_cdf


class ExponentialFitter(UnivariateFitter):

    """
    This class implements an Exponential model for univariate data. The model has parameterized
    form:

      S(t) = exp(-(lambda*t)),   lambda >0

    which implies the cumulative hazard rate is

      H(t) = lambda*t

    and the hazard rate is:

      h(t) = lambda

    After calling the `.fit` method, you have access to properties like:
     'survival_function_', 'lambda_'

    """

    def fit(self, durations, event_observed=None, timeline=None, entry=None,
            label='Exponential_estimate', alpha=None, ci_labels=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          event_observed: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated observations, i.e the birth event was not observed.
             If None, defaults to all 0 (all birth events observed.)
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>

        Returns:
          self, with new properties like 'survival_function_' and 'lambda_'.

        """

        self.durations = np.asarray(durations, dtype=float)
        self.event_observed = np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        self.timeline = np.sort(np.asarray(timeline)) if timeline is not None else np.arange(int(self.durations.min()), int(self.durations.max()) + 1)
        self._label = label

        # estimation
        D = self.event_observed.sum()
        T = self.durations.sum()
        self.lambda_ = D / T
        self._lambda_variance_ = self.lambda_ / T
        self.survival_function_ = pd.DataFrame(np.exp(-self.lambda_ * self.timeline), columns=[self._label], index=self.timeline)
        self.confidence_interval_ = self._bounds(alpha if alpha else self.alpha, ci_labels)
        self.median_ = 1. / self.lambda_ * (np.log(2))

        # estimation functions
        self.predict = self._predict("survival_function_", self._label)
        self.subtract = self._subtract("survival_function_")
        self.divide = self._divide("survival_function_")

        # plotting
        self.plot = self._plot_estimate("survival_function_")
        self.plot_survival_function_ = self.plot

        return self

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1. + alpha) / 2.)
        df = pd.DataFrame(index=self.timeline)

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        std = np.sqrt(self._lambda_variance_)
        sv = self.survival_function_
        error = std * self.timeline[:, None] * sv
        df[ci_labels[0]] = sv + alpha2 * error
        df[ci_labels[1]] = sv - alpha2 * error
        return df
