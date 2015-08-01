# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd

from lifelines.fitters import UnivariateFitter
from lifelines.utils import _preprocess_inputs, _additive_estimate, StatError, inv_normal_cdf,\
    median_survival_times


class KaplanMeierFitter(UnivariateFitter):

    """
    Class for fitting the Kaplan-Meier estimate for the survival function.

    KaplanMeierFitter( alpha=0.95)

    alpha: The alpha value associated with the confidence intervals.

    """

    def fit(self, durations, event_observed=None, timeline=None, entry=None, label='KM_estimate',
            alpha=None, left_censorship=False, ci_labels=None):
        """
        Parameters:
          duration: an array, or pd.Series, of length n -- duration subject was observed for
          timeline: return the best estimate at the values in timelines (postively increasing)
          event_observed: an array, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated (not left-censored) observations. If None, all members of the population
             were born at time 0.
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          left_censorship: True if durations and event_observed refer to left censorship events. Default False
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>


        Returns:
          self, with new properties like 'survival_function_'.

        """
        # if the user is interested in left-censorship, we return the cumulative_density_, no survival_function_,
        estimate_name = 'survival_function_' if not left_censorship else 'cumulative_density_'
        v = _preprocess_inputs(durations, event_observed, timeline, entry)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v
        self._label = label
        alpha = alpha if alpha else self.alpha
        log_survival_function, cumulative_sq_ = _additive_estimate(self.event_table, self.timeline,
                                                                   self._additive_f, self._additive_var,
                                                                   left_censorship)

        if entry is not None:
            # a serious problem with KM is that when the sample size is small and there are too few early
            # truncation times, it may happen that is the number of patients at risk and the number of deaths is the same.
            # we adjust for this using the Breslow-Fleming-Harrington estimator
            n = self.event_table.shape[0]
            net_population = (self.event_table['entrance'] - self.event_table['removed']).cumsum()
            if net_population.iloc[:int(n / 2)].min() == 0:
                ix = net_population.iloc[:int(n / 2)].argmin()
                raise StatError("""There are too few early truncation times and too many events. S(t)==0 for all t>%.1f. Recommend BreslowFlemingHarringtonFitter.""" % ix)

        # estimation
        setattr(self, estimate_name, pd.DataFrame(np.exp(log_survival_function), columns=[self._label]))
        self.__estimate = getattr(self, estimate_name)
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha, ci_labels)
        self.median_ = median_survival_times(self.__estimate)

        # estimation methods
        self.predict = self._predict(estimate_name, label)
        self.subtract = self._subtract(estimate_name)
        self.divide = self._divide(estimate_name)

        # plotting functions
        self.plot = self._plot_estimate(estimate_name)
        setattr(self, "plot_" + estimate_name, self.plot)
        return self

    def _bounds(self, cumulative_sq_, alpha, ci_labels):
        # See http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes2.pdf
        alpha2 = inv_normal_cdf((1. + alpha) / 2.)
        df = pd.DataFrame(index=self.timeline)
        v = np.log(self.__estimate.values)

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = np.exp(-np.exp(np.log(-v) + alpha2 * np.sqrt(cumulative_sq_) / v))
        df[ci_labels[1]] = np.exp(-np.exp(np.log(-v) - alpha2 * np.sqrt(cumulative_sq_) / v))
        return df

    def _additive_f(self, population, deaths):
        np.seterr(invalid='ignore')
        return (np.log(population - deaths) - np.log(population))

    def _additive_var(self, population, deaths):
        np.seterr(divide='ignore')
        return (1. * deaths / (population * (population - deaths))).replace([np.inf], 0)
