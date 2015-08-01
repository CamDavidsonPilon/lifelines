# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

from lifelines.fitters import UnivariateFitter
from lifelines.fitters.nelson_aalen_fitter import NelsonAalenFitter
from lifelines.utils import median_survival_times


class BreslowFlemingHarringtonFitter(UnivariateFitter):

    """
    Class for fitting the Breslow-Fleming-Harrington estimate for the survival function. This estimator
    is a biased estimator of the survival function but is more stable when the popualtion is small and
    there are too few early truncation times, it may happen that is the number of patients at risk and
    the number of deaths is the same.

    Mathematically, the NAF estimator is the negative logarithm of the BFH estimator.

    BreslowFlemingHarringtonFitter(alpha=0.95)

    alpha: The alpha value associated with the confidence intervals.

    """

    def fit(self, durations, event_observed=None, timeline=None, entry=None,
            label='BFH_estimate', alpha=None, ci_labels=None):
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
          self, with new properties like 'survival_function_'.

        """
        self._label = label
        alpha = alpha if alpha is not None else self.alpha

        naf = NelsonAalenFitter(alpha)
        naf.fit(durations, event_observed=event_observed, timeline=timeline, label=label, entry=entry, ci_labels=ci_labels)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = \
            naf.durations, naf.event_observed, naf.timeline, naf.entry, naf.event_table

        # estimation
        self.survival_function_ = np.exp(-naf.cumulative_hazard_)
        self.confidence_interval_ = np.exp(-naf.confidence_interval_)
        self.median_ = median_survival_times(self.survival_function_)

        # estimation methods
        self.predict = self._predict("survival_function_", label)
        self.subtract = self._subtract("survival_function_")
        self.divide = self._divide("survival_function_")

        # plotting functions
        self.plot = self._plot_estimate("survival_function_")
        self.plot_survival_function = self.plot
        return self
