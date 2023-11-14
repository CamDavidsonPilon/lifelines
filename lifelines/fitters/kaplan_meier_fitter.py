# -*- coding: utf-8 -*-
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import typing as t

from lifelines.fitters import NonParametricUnivariateFitter
from lifelines.exceptions import StatError, StatisticalWarning
from lifelines.utils import (
    _preprocess_inputs,
    _additive_estimate,
    _to_1d_array,
    inv_normal_cdf,
    median_survival_times,
    qth_survival_time,
    check_nans_or_infs,
    coalesce,
    CensoringType,
    pass_for_numeric_dtypes_or_raise_array,
    check_nans_or_infs,
)
from lifelines.plotting import loglogs_plot, _plot_estimate
from lifelines.fitters.npmle import npmle, reconstruct_survival_function, npmle_compute_confidence_intervals


class KaplanMeierFitter(NonParametricUnivariateFitter):

    """
    Class for fitting the Kaplan-Meier estimate for the survival function.

    Parameters
    ----------
    alpha: float, optional (default=0.05)
        The alpha value associated with the confidence intervals.
    label: string, optional
        Provide a new label for the estimate - useful if looking at many groups.

    Examples
    --------
    .. code:: python

        from lifelines import KaplanMeierFitter
        from lifelines.datasets import load_waltons
        waltons = load_waltons()

        kmf = KaplanMeierFitter(label="waltons_data")
        kmf.fit(waltons['T'], waltons['E'])
        kmf.plot()


    Attributes
    ----------
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    median_survival_time_ : float
        The estimated median time to event. np.inf if doesn't exist.
    confidence_interval_ : DataFrame
        The lower and upper confidence intervals for the survival function. An alias of
        ``confidence_interval_survival_function_``. Uses Greenwood's Exponential formula ("log-log" in R).
    confidence_interval_survival_function_ : DataFrame
        The lower and upper confidence intervals for the survival function. An alias of
        ``confidence_interval_``. Uses Greenwood's Exponential formula ("log-log" in R).
    cumulative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    confidence_interval_cumulative_density_ : DataFrame
        The lower and upper confidence intervals for the cumulative density.
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
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Fit the model to a right-censored dataset

        Parameters
        ----------
          durations: an array, list, pd.DataFrame or pd.Series
            length n -- duration (relative to subject's birth) the subject was alive for.
          event_observed: an array, list, pd.DataFrame, or pd.Series, optional
             True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
          timeline: an array, list, pd.DataFrame, or pd.Series, optional
            return the best estimate at the values in timelines (positively increasing)
          entry: an array, list, pd.DataFrame, or pd.Series, optional
             relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
             entered study when they were "born".
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
          ci_labels: tuple, optional
                add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
          weights: an array, list, pd.DataFrame, or pd.Series, optional
              if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.
          fit_options:
            Not used in KaplanMeierFitter

        Returns
        -------
        self: KaplanMeierFitter
          self with new properties like ``survival_function_``, ``plot()``, ``median_survival_time_``

        """

        return self._fit(durations, event_observed, timeline, entry, label, alpha, ci_labels, weights)

    @CensoringType.interval_censoring
    def fit_interval_censoring(
        self,
        lower_bound,
        upper_bound,
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        entry=None,
        weights=None,
        tol: float = 1e-5,
        show_progress: bool = False,
        fit_options=None,
        **kwargs,
    ) -> KaplanMeierFitter:
        """
        Fit the model to a interval-censored dataset using non-parametric MLE. This estimator is
        also called the Turnbull Estimator.

        Currently, only closed interval are supported. However, it's easy to create open intervals by adding (or subtracting) a very small
        value from the lower-bound (or upper bound). For example, the following turns closed intervals into open intervals.

        >>> left, right = df['left'], df['right']
        >>> KaplanMeierFitter().fit_interval_censoring(left + 0.00001, right - 0.00001)

        Note
        ------
        This is new and experimental, and many features are missing.

        Parameters
        ----------
          lower_bound: an array, list, pd.DataFrame or pd.Series
            length n -- lower bound of observations
          upper_bound: an array, list, pd.DataFrame or pd.Series
            length n -- upper bound of observations
          event_observed: an array, list, pd.DataFrame, or pd.Series, optional
             True if the the death was observed, False if the event was lost (right-censored). This can be computed from
             the lower_bound and upper_bound, and can be left blank.
          timeline: an array, list, pd.DataFrame, or pd.Series, optional
            return the best estimate at the values in timelines (positively increasing)
          entry: an array, list, pd.DataFrame, or pd.Series, optional
             relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
             entered study when they were "born".
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
          ci_labels: tuple, optional
                add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
          weights: an array, list, pd.DataFrame, or pd.Series, optional
              if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.
          tol: float, optional
            minimum difference in log likelihood changes for iterative algorithm.
          show_progress: bool, optional
            display information during fitting.

        Returns
        -------
        self: KaplanMeierFitter
          self with new properties like ``survival_function_``, ``plot()``, ``median_survival_time_``
        """
        if entry is not None:
            raise NotImplementedError("entry is not supported yet")

        if weights is None:
            weights = np.ones_like(upper_bound)

        self.weights = np.asarray(weights)

        self.upper_bound = np.atleast_1d(pass_for_numeric_dtypes_or_raise_array(upper_bound))
        self.lower_bound = np.atleast_1d(pass_for_numeric_dtypes_or_raise_array(lower_bound))
        check_nans_or_infs(self.lower_bound)

        self.event_observed = self.lower_bound == self.upper_bound

        self.timeline = coalesce(timeline, np.unique(np.concatenate((self.upper_bound, self.lower_bound))))

        if (self.upper_bound < self.lower_bound).any():
            raise ValueError("All upper_bound times must be greater than or equal to lower_bound times.")

        if event_observed is None:
            event_observed = self.upper_bound == self.lower_bound

        if ((self.lower_bound == self.upper_bound) != event_observed).any():
            raise ValueError(
                "For all rows, lower_bound == upper_bound if and only if event observed = 1 (uncensored). Likewise, lower_bound < upper_bound if and only if event observed = 0 (censored)"
            )

        self._label = coalesce(label, self._label, "NPMLE_estimate")

        results = npmle(self.lower_bound, self.upper_bound, verbose=show_progress, tol=tol, weights=weights, **kwargs)
        self.survival_function_ = reconstruct_survival_function(*results, self.timeline, label=self._label).loc[self.timeline]
        self.cumulative_density_ = 1 - self.survival_function_

        self._median = median_survival_times(self.survival_function_)

        """
        self.confidence_interval_ = npmle_compute_confidence_intervals(self.lower_bound, self.upper_bound, self.survival_function_, self.alpha)
        self.confidence_interval_survival_function_ = self.confidence_interval_
        self.confidence_interval_cumulative_density_ = 1 - self.confidence_interval_
        """
        # estimation methods
        self._estimation_method = "survival_function_"
        self._estimate_name = "survival_function_"
        return self

    @CensoringType.left_censoring
    def fit_left_censoring(
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
    ):
        """
        Fit the model to a left-censored dataset

        Parameters
        ----------
          durations: an array, list, pd.DataFrame or pd.Series
            length n -- duration subject was observed for
          event_observed: an array, list, pd.DataFrame, or pd.Series, optional
             True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
          timeline: an array, list, pd.DataFrame, or pd.Series, optional
            return the best estimate at the values in timelines (positively increasing)
          entry: an array, list, pd.DataFrame, or pd.Series, optional
             relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
             entered study when they were "born".
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
          ci_labels: tuple, optional
                add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
          weights: an array, list, pd.DataFrame, or pd.Series, optional
              if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns
        -------
        self: KaplanMeierFitter
          self with new properties like ``survival_function_``, ``plot()``, ``median_survival_time_``

        """
        # left censoring is then defined in CensoringType.is_left_censoring(self)
        return self._fit(durations, event_observed, timeline, entry, label, alpha, ci_labels, weights)

    def _fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        entry=None,
        label: t.Optional[str] = None,
        alpha=None,
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
            return the best estimate at the values in timelines (positively increasing)
          entry: an array, list, pd.DataFrame, or pd.Series, optional
             relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
             entered study when they were "born".
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing alpha for this call to fit only.
          ci_labels: tuple, optional
                add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
          weights: an array, list, pd.DataFrame, or pd.Series, optional
              if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns
        -------
        self: KaplanMeierFitter
          self with new properties like ``survival_function_``, ``plot()``, ``median_survival_time_``

        """
        durations = np.asarray(durations)
        self._check_values(durations)

        if event_observed is not None:
            event_observed = np.asarray(event_observed)
            self._check_values(event_observed)

        self._label = coalesce(label, self._label, "KM_estimate")

        if weights is not None:
            weights = np.asarray(weights)
            if (weights.astype(int) != weights).any():
                warnings.warn(
                    """It looks like your weights are not integers, possibly propensity scores then?
  It's important to know that the naive variance estimates of the coefficients are biased. Instead use Monte Carlo to
  estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
  or "Adjusted Kaplan-Meier estimator and log-rank test with inverse probability of treatment weighting for survival data."
                  """,
                    StatisticalWarning,
                )
        else:
            weights = np.ones_like(durations, dtype=float)

        # if the user is interested in left-censorship, we return estimate the cumulative density, not survival function
        is_left_censoring = CensoringType.is_left_censoring(self)
        primary_estimate_name = "survival_function_"
        secondary_estimate_name = "cumulative_density_"

        (
            self.durations,
            self.event_observed,
            self.timeline,
            self.entry,
            self.event_table,
            self.weights,
        ) = _preprocess_inputs(durations, event_observed, timeline, entry, weights)

        alpha = alpha if alpha else self.alpha
        log_estimate, cumulative_sq_ = _additive_estimate(
            self.event_table, self.timeline, self._additive_f, self._additive_var, is_left_censoring
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
        if is_left_censoring:
            self.cumulative_density_ = pd.DataFrame(np.exp(log_estimate), columns=[self._label])
            self.survival_function_ = 1 - self.cumulative_density_
            self.confidence_interval_ = 1 - self._bounds(self.cumulative_density_, cumulative_sq_, alpha, ci_labels)

        else:
            self.survival_function_ = pd.DataFrame(np.exp(log_estimate), columns=[self._label])
            self.cumulative_density_ = 1 - self.survival_function_
            self.confidence_interval_ = self._bounds(self.survival_function_, cumulative_sq_, alpha, ci_labels)

        self.confidence_interval_survival_function_ = self.confidence_interval_
        self.confidence_interval_cumulative_density_ = 1 - self.confidence_interval_
        self.confidence_interval_cumulative_density_[:] = np.fliplr(self.confidence_interval_cumulative_density_.values)
        self._median = median_survival_times(self.survival_function_)
        self._cumulative_sq_ = cumulative_sq_

        # estimation methods
        self._estimation_method = "survival_function_"
        self._estimate_name = "survival_function_"

        return self

    @property
    def median_survival_time_(self) -> float:
        return self._median

    def _check_values(self, array):
        check_nans_or_infs(array)

    def plot_loglogs(self, *args, **kwargs):
        r"""
        Plot :math:`\log(-\log(S(t)))` against :math:`\log(t)`. Same arguments as ``.plot``.
        """
        return loglogs_plot(self, *args, **kwargs)

    def survival_function_at_times(self, times, label=None) -> pd.Series:
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
        return pd.Series(self.predict(times), index=_to_1d_array(times), name=label)

    def cumulative_density_at_times(self, times, label=None) -> pd.Series:
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
        return pd.Series(1 - self.predict(times), index=_to_1d_array(times), name=label)

    def plot(self, **kwargs):
        warnings.warn(
            "The `plot` function is deprecated, and will be removed in future versions. Use `plot_survival_function`",
            DeprecationWarning,
        )
        return self.plot_survival_function(**kwargs)

    def plot_survival_function(self, **kwargs):
        """Alias of ``plot``"""
        if not CensoringType.is_interval_censoring(self):
            return _plot_estimate(self, estimate="survival_function_", **kwargs)
        else:
            # hack for now.
            def safe_pop(dict, key):
                if key in dict:
                    return dict.pop(key)
                else:
                    return None

            color = coalesce(safe_pop(kwargs, "c"), safe_pop(kwargs, "color"), "k")
            self.survival_function_.plot(drawstyle="steps-pre", color=color, **kwargs)

    def plot_cumulative_density(self, **kwargs):
        """
        Plots a pretty figure of the cumulative density function.

        Matplotlib plot arguments can be passed in inside the kwargs.

        Parameters
        -----------
        show_censors: bool
            place markers at censorship events. Default: False
        censor_styles: bool
            If show_censors, this dictionary will be passed into the plot call.
        ci_alpha: bool
            the transparency level of the confidence interval. Default: 0.3
        ci_force_lines: bool
            force the confidence intervals to be line plots (versus default shaded areas). Default: False
        ci_show: bool
            show confidence intervals. Default: True
        ci_legend: bool
            if ci_force_lines is True, this is a boolean flag to add the lines' labels to the legend. Default: False
        at_risk_counts: bool
            show group sizes at time points. See function ``add_at_risk_counts`` for details. Default: False
        loc: slice
            specify a time-based subsection of the curves to plot, ex:

            >>> model.plot(loc=slice(0.,10.))

            will plot the time values between t=0. and t=10.
        iloc: slice
            specify a location-based subsection of the curves to plot, ex:

            >>> model.plot(iloc=slice(0,10))

            will plot the first 10 time points.

        Returns
        -------
        ax:
            a pyplot axis object
        """
        if not CensoringType.is_interval_censoring(self):
            return _plot_estimate(self, estimate="cumulative_density_", **kwargs)
        else:
            # hack for now.
            color = coalesce(kwargs.get("c"), kwargs.get("color"), "k")
            self.cumulative_density_.plot(drawstyle="steps", color=color, **kwargs)

    def _bounds(self, estimate_, cumulative_sq_, alpha, ci_labels):
        # This method calculates confidence intervals using the exponential Greenwood formula.
        # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
        z = inv_normal_cdf(1 - alpha / 2)
        df = pd.DataFrame(index=self.timeline)
        v = np.log(estimate_.values)

        if ci_labels is None:
            ci_labels = ["%s_lower_%g" % (self._label, 1 - alpha), "%s_upper_%g" % (self._label, 1 - alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = np.exp(-np.exp(np.log(-v) - z * np.sqrt(cumulative_sq_.values[:, None]) / v))
        df[ci_labels[1]] = np.exp(-np.exp(np.log(-v) + z * np.sqrt(cumulative_sq_.values[:, None]) / v))
        return df.fillna(1.0)

    def _additive_f(self, population, deaths):
        np.seterr(invalid="ignore", divide="ignore")
        return np.log(population - deaths) - np.log(population)

    def _additive_var(self, population, deaths):
        np.seterr(divide="ignore")
        population = population.astype("uint64")
        return (deaths / (population * (population - deaths))).replace([np.inf], 0)

    def plot_cumulative_hazard(self, **kwargs):
        raise NotImplementedError(
            "The Kaplan-Meier estimator is not used to estimate the cumulative hazard. Try the NelsonAalenFitter or any other parametric model"
        )

    def plot_hazard(self, **kwargs):
        raise NotImplementedError(
            "The Kaplan-Meier estimator is not used to estimate the hazard. Try the NelsonAalenFitter or any other parametric model"
        )
