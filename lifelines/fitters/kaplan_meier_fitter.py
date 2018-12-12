# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import warnings
import numpy as np
import pandas as pd

from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    _preprocess_inputs,
    _additive_estimate,
    StatError,
    inv_normal_cdf,
    median_survival_times,
    check_nans_or_infs,
)
from lifelines.plotting import plot_loglogs


class KaplanMeierFitter(UnivariateFitter):

    """
    Class for fitting the Kaplan-Meier estimate for the survival function.

    Parameters
    ----------
    alpha: float, option (default=0.95)
        The alpha value associated with the confidence intervals.

    Examples
    --------
    >>> from lifelines import KaplanMeierFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> kmf = KaplanMeierFitter()
    >>> kmf.fit(waltons['T'], waltons['E'])
    >>> kmf.plot()
    """

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label="KM_estimate",
        alpha=None,
        left_censorship=False,
        ci_labels=None,
        weights=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Parameters
        ----------
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
          weights: n array, or pd.Series, of length n, if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns
        -------
        self: KaplanMeierFitter
          self with new properties like 'survival_function_'.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        if weights is not None:
            if (weights.astype(int) != weights).any():
                warnings.warn(
                    """It looks like your weights are not integers, possibly prospenity scores then?
  It's important to know that the naive variance estimates of the coefficients are biased. Instead use Monte Carlo to
  estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
  or "Adjusted Kaplan-Meier estimator and log-rank test with inverse probability of treatment weighting for survival data."
                  """,
                    RuntimeWarning,
                )

        # if the user is interested in left-censorship, we return the cumulative_density_, no survival_function_,
        estimate_name = "survival_function_" if not left_censorship else "cumulative_density_"
        v = _preprocess_inputs(durations, event_observed, timeline, entry, weights)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v

        self._label = label
        alpha = alpha if alpha else self.alpha
        log_survival_function, cumulative_sq_ = _additive_estimate(
            self.event_table, self.timeline, self._additive_f, self._additive_var, left_censorship
        )

        if entry is not None:
            # a serious problem with KM is that when the sample size is small and there are too few early
            # truncation times, it may happen that is the number of patients at risk and the number of deaths is the same.
            # we adjust for this using the Breslow-Fleming-Harrington estimator
            n = self.event_table.shape[0]
            net_population = (self.event_table["entrance"] - self.event_table["removed"]).cumsum()
            if net_population.iloc[: int(n / 2)].min() == 0:
                ix = net_population.iloc[: int(n / 2)].idxmin()
                raise StatError(
                    """There are too few early truncation times and too many events. S(t)==0 for all t>%.1f. Recommend BreslowFlemingHarringtonFitter."""
                    % ix
                )

        # estimation
        setattr(self, estimate_name, pd.DataFrame(np.exp(log_survival_function), columns=[self._label]))
        self.__estimate = getattr(self, estimate_name)
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha, ci_labels)
        self.median_ = median_survival_times(self.__estimate, left_censorship=left_censorship)

        # estimation methods
        self._estimation_method = estimate_name
        self._estimate_name = estimate_name
        self._predict_label = label
        self._update_docstrings()

        # plotting functions
        setattr(self, "plot_" + estimate_name, self.plot)
        return self

    def plot_loglogs(self, *args, **kwargs):
        return plot_loglogs(self, *args, **kwargs)

    def _bounds(self, cumulative_sq_, alpha, ci_labels):
        # This method calculates confidence intervals using the exponential Greenwood formula.
        # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf

        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)
        v = np.log(self.__estimate.values)

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = np.exp(-np.exp(np.log(-v) + alpha2 * np.sqrt(cumulative_sq_) / v))
        df[ci_labels[1]] = np.exp(-np.exp(np.log(-v) - alpha2 * np.sqrt(cumulative_sq_) / v))
        return df

    def _additive_f(self, population, deaths):
        np.seterr(invalid="ignore", divide="ignore")
        return np.log(population - deaths) - np.log(population)

    def _additive_var(self, population, deaths):
        np.seterr(divide="ignore")
        return (1.0 * deaths / (population * (population - deaths))).replace([np.inf], 0)
