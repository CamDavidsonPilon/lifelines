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
    epanechnikov_kernel,
    inv_normal_cdf,
    check_nans_or_infs,
)


class NelsonAalenFitter(UnivariateFitter):

    """
    Class for fitting the Nelson-Aalen estimate for the cumulative hazard.

    NelsonAalenFitter( alpha=0.95, nelson_aalen_smoothing=True)

    alpha: The alpha value associated with the confidence intervals.
    nelson_aalen_smoothing: If the event times are naturally discrete (like discrete years, minutes, etc.)
      then it is advisable to turn this parameter to False. See [1], pg.84.

    [1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

    """

    def __init__(self, alpha=0.95, nelson_aalen_smoothing=True):
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha parameter must be between 0 and 1.")
        self.alpha = alpha
        self.nelson_aalen_smoothing = nelson_aalen_smoothing

        if self.nelson_aalen_smoothing:
            self._variance_f = self._variance_f_smooth
            self._additive_f = self._additive_f_smooth
        else:
            self._variance_f = self._variance_f_discrete
            self._additive_f = self._additive_f_discrete

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label="NA_estimate",
        alpha=None,
        ci_labels=None,
        weights=None,
    ):  # pylint: disable=too-many-arguments
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
          weights: n array, or pd.Series, of length n, if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns:
          self, with new properties like 'cumulative_hazard_'.

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

        v = _preprocess_inputs(durations, event_observed, timeline, entry, weights)
        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = v

        cumulative_hazard_, cumulative_sq_ = _additive_estimate(
            self.event_table, self.timeline, self._additive_f, self._variance_f, False
        )

        # esimates
        self._label = label
        self.cumulative_hazard_ = pd.DataFrame(cumulative_hazard_, columns=[self._label])
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha if alpha else self.alpha, ci_labels)
        self._cumulative_sq = cumulative_sq_

        # estimation methods
        self._estimation_method = "cumulative_hazard_"
        self._estimate_name = "cumulative_hazard_"
        self._predict_label = label
        self._update_docstrings()

        # plotting
        self.plot_cumulative_hazard = self.plot

        return self

    def plot_hazard(self, *args, **kwargs):
        kwargs["estimate"] = "hazard_"
        return self.plot(*args, **kwargs)

    def _bounds(self, cumulative_sq_, alpha, ci_labels):
        alpha2 = inv_normal_cdf(1 - (1 - alpha) / 2)
        df = pd.DataFrame(index=self.timeline)

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."
        self.ci_labels = ci_labels

        cum_hazard_ = self.cumulative_hazard_.values
        df[ci_labels[0]] = cum_hazard_ * np.exp(
            alpha2 * np.sqrt(cumulative_sq_) / np.where(cum_hazard_ == 0, 1, cum_hazard_)
        )
        df[ci_labels[1]] = cum_hazard_ * np.exp(
            -alpha2 * np.sqrt(cumulative_sq_) / np.where(cum_hazard_ == 0, 1, cum_hazard_)
        )
        return df

    def _variance_f_smooth(self, population, deaths):
        cum_ = np.cumsum(1.0 / np.arange(1, np.max(population) + 1) ** 2)
        return pd.Series(
            cum_[population - 1] - np.where(population - deaths - 1 >= 0, cum_[population - deaths - 1], 0),
            index=population.index,
        )

    def _variance_f_discrete(self, population, deaths):
        return 1.0 * (population - deaths) * deaths / population ** 3

    def _additive_f_smooth(self, population, deaths):
        cum_ = np.cumsum(1.0 / np.arange(1, np.max(population) + 1))
        return pd.Series(
            cum_[population - 1] - np.where(population - deaths - 1 >= 0, cum_[population - deaths - 1], 0),
            index=population.index,
        )

    def _additive_f_discrete(self, population, deaths):
        return (1.0 * deaths / population).replace([np.inf], 0)

    def smoothed_hazard_(self, bandwidth):
        """
        Parameters:
          bandwidth: the bandwith used in the Epanechnikov kernel.

        Returns:
          a DataFrame of the smoothed hazard
        """
        timeline = self.timeline
        cumulative_hazard_name = self.cumulative_hazard_.columns[0]
        hazard_name = "differenced-" + cumulative_hazard_name
        hazard_ = self.cumulative_hazard_.diff().fillna(self.cumulative_hazard_.iloc[0])
        C = (hazard_[cumulative_hazard_name] != 0.0).values
        return pd.DataFrame(
            1.0
            / bandwidth
            * np.dot(epanechnikov_kernel(timeline[:, None], timeline[C][None, :], bandwidth), hazard_.values[C, :]),
            columns=[hazard_name],
            index=timeline,
        )

    def smoothed_hazard_confidence_intervals_(self, bandwidth, hazard_=None):
        """
        Parameter:
          bandwidth: the bandwith to use in the Epanechnikov kernel.
          hazard_: a computed (n,) numpy array of estimated hazard rates. If none, uses naf.smoothed_hazard_
        """
        if hazard_ is None:
            hazard_ = self.smoothed_hazard_(bandwidth).values[:, 0]

        timeline = self.timeline
        alpha2 = inv_normal_cdf(1 - (1 - self.alpha) / 2)
        self._cumulative_sq.iloc[0] = 0
        var_hazard_ = self._cumulative_sq.diff().fillna(self._cumulative_sq.iloc[0])
        C = var_hazard_.values != 0.0  # only consider the points with jumps
        std_hazard_ = np.sqrt(
            1.0
            / (bandwidth ** 2)
            * np.dot(
                epanechnikov_kernel(timeline[:, None], timeline[C][None, :], bandwidth) ** 2, var_hazard_.values[C]
            )
        )
        values = {
            self.ci_labels[0]: hazard_ * np.exp(alpha2 * std_hazard_ / hazard_),
            self.ci_labels[1]: hazard_ * np.exp(-alpha2 * std_hazard_ / hazard_),
        }
        return pd.DataFrame(values, index=timeline)

    @property
    def conditional_time_to_event_(self):
        raise NotImplementedError
