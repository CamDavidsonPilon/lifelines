# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd

from lifelines.fitters import UnivariateFitter
from lifelines.plotting import _plot_estimate
from lifelines.exceptions import StatisticalWarning
from lifelines.utils import (
    _preprocess_inputs,
    _additive_estimate,
    epanechnikov_kernel,
    inv_normal_cdf,
    check_nans_or_infs,
    CensoringType,
    coalesce,
    _to_1d_array,
)


class NelsonAalenFitter(UnivariateFitter):

    """
    Class for fitting the Nelson-Aalen estimate for the cumulative hazard.

    NelsonAalenFitter(alpha=0.05, nelson_aalen_smoothing=True)

    Parameters
    ----------
    alpha: float, optional (default=0.05)
        The alpha value associated with the confidence intervals.
    nelson_aalen_smoothing: bool, optional
        If the event times are naturally discrete (like discrete years, minutes, etc.)
      then it is advisable to turn this parameter to False. See [1], pg.84.

    Notes
    ------
    [1] Aalen, O., Borgan, O., Gjessing, H., 2008. Survival and Event History Analysis

    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    confidence_interval_ : DataFrame
        The lower and upper confidence intervals for the cumulative hazard
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    event_table: DataFrame
        A summary of the life table

    """

    def __init__(self, alpha=0.05, nelson_aalen_smoothing=True, **kwargs):
        super(NelsonAalenFitter, self).__init__(alpha=alpha, **kwargs)
        self.alpha = alpha
        self.nelson_aalen_smoothing = nelson_aalen_smoothing

        if self.nelson_aalen_smoothing:
            self._variance_f = self._variance_f_smooth
            self._additive_f = self._additive_f_smooth
        else:
            self._variance_f = self._variance_f_discrete
            self._additive_f = self._additive_f_discrete

    @CensoringType.right_censoring
    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label=None,
        alpha=None,
        ci_labels=None,
        weights=None,
        fit_options=None,
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        -----------
        durations: an array, or pd.Series, of length n
          duration subject was observed for
        timeline: iterable
            return the best estimate at the values in timelines (positively increasing)
        event_observed: an array, or pd.Series, of length n
            True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
        entry: an array, or pd.Series, of length n
           relative time when a subject entered the study. This is
           useful for left-truncated observations, i.e the birth event was not observed.
           If None, defaults to all 0 (all birth events observed.)
        label: string
            a string to name the column of the estimate.
        alpha: float
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: iterable
            add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
        weights: n array, or pd.Series, of length n
            if providing a weighted dataset. For example, instead
            of providing every subject as a single element of `durations` and `event_observed`, one could
            weigh subject differently.
        fit_options:
            Not used

        Returns
        -------
          self, with new properties like ``cumulative_hazard_``.

        """
        durations = np.asarray(durations)
        check_nans_or_infs(durations)
        if event_observed is not None:
            event_observed = np.asarray(event_observed)
            check_nans_or_infs(event_observed)

        if weights is not None:
            if (weights.astype(int) != weights).any():
                warnings.warn(
                    """It looks like your weights are not integers, possibly propensity scores then?
  It's important to know that the naive variance estimates of the coefficients are biased. Instead use Monte Carlo to
  estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
  or "Adjusted Kaplan-Meier estimator and log-rank test with inverse probability of treatment weighting for survival data."
                  """,
                    StatisticalWarning,
                )

        (self.durations, self.event_observed, self.timeline, self.entry, self.event_table, self.weights) = _preprocess_inputs(
            durations, event_observed, timeline, entry, weights
        )

        cumulative_hazard_, cumulative_sq_ = _additive_estimate(
            self.event_table, self.timeline, self._additive_f, self._variance_f, False
        )

        # estimates
        self._label = coalesce(label, self._label, "NA_estimate")
        self.cumulative_hazard_ = pd.DataFrame(cumulative_hazard_, columns=[self._label])
        self.confidence_interval_ = self._bounds(cumulative_sq_.values[:, None], alpha if alpha else self.alpha, ci_labels)
        self.confidence_interval_cumulative_hazard_ = self.confidence_interval_
        self._cumulative_sq = cumulative_sq_

        # estimation methods
        self._estimation_method = "cumulative_hazard_"
        self._estimate_name = "cumulative_hazard_"

        # plotting
        self.plot_cumulative_hazard = self.plot

        return self

    def plot_hazard(self, bandwidth=None, **kwargs):
        if bandwidth is None:
            raise ValueError("Must specify a bandwidth parameter in the call to plot_hazard, e.g. `plot_hazard(bandwidth=1.0)`")
        estimate = self.smoothed_hazard_(bandwidth)
        confidence_intervals = self.smoothed_hazard_confidence_intervals_(bandwidth, estimate.values[:, 0])
        return _plot_estimate(self, estimate, confidence_intervals=confidence_intervals, **kwargs)

    def _bounds(self, cumulative_sq_, alpha, ci_labels):
        z = inv_normal_cdf(1 - alpha / 2)
        df = pd.DataFrame(index=self.timeline)

        if ci_labels is None:
            ci_labels = ["%s_lower_%g" % (self._label, 1 - alpha), "%s_upper_%g" % (self._label, 1 - alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."
        self.ci_labels = ci_labels

        cum_hazard_ = self.cumulative_hazard_.values
        df[ci_labels[0]] = cum_hazard_ * np.exp(-z * np.sqrt(cumulative_sq_) / np.where(cum_hazard_ == 0, 1, cum_hazard_))
        df[ci_labels[1]] = cum_hazard_ * np.exp(z * np.sqrt(cumulative_sq_) / np.where(cum_hazard_ == 0, 1, cum_hazard_))
        return df

    def _variance_f_smooth(self, population, deaths):
        cum_ = np.cumsum(1.0 / np.arange(1, np.max(population) + 1) ** 2)
        return pd.Series(
            cum_[population - 1] - np.where(population - deaths - 1 >= 0, cum_[population - deaths - 1], 0),
            index=population.index,
        )

    def _variance_f_discrete(self, population, deaths):
        return (1 - deaths / population) * (deaths / population) * (1.0 / population)

    def _additive_f_smooth(self, population, deaths):
        cum_ = np.cumsum(1.0 / np.arange(1, np.max(population) + 1))
        return pd.Series(
            cum_[population - 1] - np.where(population - deaths - 1 >= 0, cum_[population - deaths - 1], 0),
            index=population.index,
        )

    def _additive_f_discrete(self, population, deaths):
        return (deaths / population).replace([np.inf], 0)

    def smoothed_hazard_(self, bandwidth):
        """
        Parameters
        -----------
        bandwidth: float
            the bandwidth used in the Epanechnikov kernel.

        Returns
        -------
        DataFrame:
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
        Parameters
        ----------
          bandwidth: float
            the bandwidth to use in the Epanechnikov kernel. > 0
          hazard_: numpy array
            a computed (n,) numpy array of estimated hazard rates. If none, uses ``smoothed_hazard_``
        """
        if hazard_ is None:
            hazard_ = self.smoothed_hazard_(bandwidth).values[:, 0]

        timeline = self.timeline
        z = inv_normal_cdf(1 - self.alpha / 2)
        self._cumulative_sq.iloc[0] = 0
        var_hazard_ = self._cumulative_sq.diff().fillna(self._cumulative_sq.iloc[0])
        C = var_hazard_.values != 0.0  # only consider the points with jumps
        std_hazard_ = np.sqrt(
            1.0
            / (bandwidth**2)
            * np.dot(epanechnikov_kernel(timeline[:, None], timeline[C][None, :], bandwidth) ** 2, var_hazard_.values[C])
        )
        values = {
            self.ci_labels[0]: hazard_ * np.exp(z * std_hazard_ / hazard_),
            self.ci_labels[1]: hazard_ * np.exp(-z * std_hazard_ / hazard_),
        }
        return pd.DataFrame(values, index=timeline)

    @property
    def conditional_time_to_event_(self):
        raise NotImplementedError()

    def percentile(self, p):
        raise NotImplementedError()

    def cumulative_hazard_at_times(self, times, label=None) -> pd.Series:
        """
        Return a Pandas series of the predicted cumhaz value at specific times

        Parameters
        -----------
        times: iterable or float

        Returns
        --------
        pd.Series

        """
        label = coalesce(label, self._label)
        return pd.Series(self.predict(times), index=_to_1d_array(times), name=label)
