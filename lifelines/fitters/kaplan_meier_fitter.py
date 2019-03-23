# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd

from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    _preprocess_inputs,
    _additive_estimate,
    _to_array,
    StatError,
    inv_normal_cdf,
    median_survival_times,
    check_nans_or_infs,
    StatisticalWarning,
    coalesce,
)
from lifelines.plotting import plot_loglogs, _plot_estimate


class KaplanMeierFitter(UnivariateFitter):

    """
    Class for fitting the Kaplan-Meier estimate for the survival function.

    Parameters
    ----------
    alpha: float, option (default=0.05)
        The alpha value associated with the confidence intervals.


    Examples
    --------
    >>> from lifelines import KaplanMeierFitter
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> kmf = KaplanMeierFitter()
    >>> kmf.fit(waltons['T'], waltons['E'])
    >>> kmf.plot()


    Attributes
    ----------
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    median_ : float
        The estimated median time to event. np.inf if doesn't exist.
    confidence_interval_ : DataFrame
        The lower and upper confidence intervals for the survival function. An alias of
        ``confidence_interval_survival_function_``
    confidence_interval_survival_function_ : DataFrame
        The lower and upper confidence intervals for the survival function. An alias of
        ``confidence_interval_``
    cumumlative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    confidence_interval_cumumlative_density_ : DataFrame
        The lower and upper confidence intervals for the cumulative density
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
          durations: an array, list, pd.DataFrame or pd.Series
            length n -- duration subject was observed for
          event_observed: an array, list, pd.DataFrame, or pd.Series, optional
             True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
          timeline: an array, list, pd.DataFrame, or pd.Series, optional
            return the best estimate at the values in timelines (postively increasing)
          entry: an array, list, pd.DataFrame, or pd.Series, optional
             relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
             entered study when they were "born".
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
          left_censorship: bool, optional (default=False)
            True if durations and event_observed refer to left censorship events. Default False
          ci_labels: tuple, optional
                add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
          weights: an array, list, pd.DataFrame, or pd.Series, optional
              if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns
        -------
        self: KaplanMeierFitter
          self with new properties like ``survival_function_``, ``plot()``, ``median``

        """

        self._check_values(durations)
        if event_observed is not None:
            self._check_values(event_observed)

        self.left_censorship = left_censorship
        self._label = label

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

        # if the user is interested in left-censorship, we return the cumulative_density_, no survival_function_,
        primary_estimate_name = "survival_function_" if not self.left_censorship else "cumulative_density_"
        secondary_estimate_name = "cumulative_density_" if not self.left_censorship else "survival_function_"

        self.durations, self.event_observed, self.timeline, self.entry, self.event_table = _preprocess_inputs(
            durations, event_observed, timeline, entry, weights
        )

        alpha = alpha if alpha else self.alpha
        log_estimate, cumulative_sq_ = _additive_estimate(
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
                    """There are too few early truncation times and too many events. S(t)==0 for all t>%g. Recommend BreslowFlemingHarringtonFitter."""
                    % ix
                )

        # estimation
        setattr(self, primary_estimate_name, pd.DataFrame(np.exp(log_estimate), columns=[self._label]))
        setattr(self, secondary_estimate_name, pd.DataFrame(1 - np.exp(log_estimate), columns=[self._label]))

        self.__estimate = getattr(self, primary_estimate_name)
        self.confidence_interval_ = self._bounds(cumulative_sq_[:, None], alpha, ci_labels)
        self.median_ = median_survival_times(self.__estimate, left_censorship=left_censorship)
        self._cumulative_sq_ = cumulative_sq_

        setattr(self, "confidence_interval_" + primary_estimate_name, self.confidence_interval_)
        setattr(self, "confidence_interval_" + secondary_estimate_name, 1 - self.confidence_interval_)

        # estimation methods
        self._estimation_method = primary_estimate_name
        self._estimate_name = primary_estimate_name
        self._predict_label = label
        self._update_docstrings()

        return self

    def _check_values(self, array):
        check_nans_or_infs(array)

    def plot_loglogs(self, *args, **kwargs):
        r"""
        Plot :math:`\log(S(t))` against :math:`\log(t)`
        """
        return plot_loglogs(self, *args, **kwargs)

    def survival_function_at_times(self, times, label=None):
        """
        Return a Pandas series of the predicted survival value at specific times

        Parameters
        -----------
        times: iterable or float

        Returns
        --------
        pd.Series

        """
        label = coalesce(label, self._label)
        return pd.Series(self.predict(times), index=_to_array(times), name=label)

    def cumulative_density_at_times(self, times, label=None):
        """
        Return a Pandas series of the predicted cumulative density at specific times

        Parameters
        -----------
        times: iterable or float

        Returns
        --------
        pd.Series

        """
        label = coalesce(label, self._label)
        return pd.Series(1 - self.predict(times), index=_to_array(times), name=label)

    def plot_survival_function(self, **kwargs):
        return _plot_estimate(
            self,
            estimate=self.survival_function_,
            confidence_intervals=self.confidence_interval_survival_function_,
            **kwargs
        )

    def plot_cumulative_density(self, **kwargs):
        return _plot_estimate(
            self,
            estimate=self.cumulative_density_,
            confidence_intervals=self.confidence_interval_cumulative_density_,
            **kwargs
        )

    def _bounds(self, cumulative_sq_, alpha, ci_labels):
        # This method calculates confidence intervals using the exponential Greenwood formula.
        # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
        z = inv_normal_cdf(1 - alpha / 2)
        df = pd.DataFrame(index=self.timeline)
        v = np.log(self.__estimate.values)

        if ci_labels is None:
            ci_labels = ["%s_upper_%g" % (self._label, 1 - alpha), "%s_lower_%g" % (self._label, 1 - alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = np.exp(-np.exp(np.log(-v) + z * np.sqrt(cumulative_sq_) / v))
        df[ci_labels[1]] = np.exp(-np.exp(np.log(-v) - z * np.sqrt(cumulative_sq_) / v))
        return df

    def _additive_f(self, population, deaths):
        np.seterr(invalid="ignore", divide="ignore")
        return np.log(population - deaths) - np.log(population)

    def _additive_var(self, population, deaths):
        np.seterr(divide="ignore")
        return (deaths / (population * (population - deaths))).replace([np.inf], 0)

    def plot_cumulative_hazard(self, **kwargs):
        raise NotImplementedError(
            "The Kaplan-Meier estimator is not used to estimate the cumulative hazard. Try the NelsonAalenFitter or any other parametric model"
        )

    def plot_hazard(self, **kwargs):
        raise NotImplementedError(
            "The Kaplan-Meier estimator is not used to estimate the hazard. Try the NelsonAalenFitter or any other parametric model"
        )
