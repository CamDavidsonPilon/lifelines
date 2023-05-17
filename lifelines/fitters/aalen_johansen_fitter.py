# -*- coding: utf-8 -*-
from textwrap import dedent
import numpy as np
import pandas as pd
import warnings

from lifelines.fitters import NonParametricUnivariateFitter
from lifelines.utils import _preprocess_inputs, inv_normal_cdf, CensoringType, coalesce
from lifelines import KaplanMeierFitter
from lifelines.plotting import _plot_estimate


class AalenJohansenFitter(NonParametricUnivariateFitter):
    """Class for fitting the Aalen-Johansen estimate for the cumulative incidence function in a competing risks framework.
    Treating competing risks as censoring can result in over-estimated cumulative density functions. Using the Kaplan
    Meier estimator with competing risks as censored is akin to estimating the cumulative density if all competing risks
    had been prevented.

    Aalen-Johansen cannot deal with tied times. We can get around this by randomly jittering the event times
    slightly. This will be done automatically and generates a warning.


    Parameters
    ----------
    alpha: float, option (default=0.05)
        The alpha value associated with the confidence intervals.

    jitter_level: float, option (default=0.00001)
        If tied event times are detected, event times are randomly changed by this factor.

    seed: int, option (default=None)
        To produce replicate results with tied event times, the numpy.random.seed can be specified in the function.

    calculate_variance: bool, option (default=True)
        By default, AalenJohansenFitter calculates the variance and corresponding confidence intervals. Due to how the
        variance is calculated, the variance must be calculated for each event time individually. This is
        computationally intensive. For some procedures, like bootstrapping, the variance is not necessary. To reduce
        computation time during these procedures, `calculate_variance` can be set to `False` to skip the variance
        calculation.

    Example
    -------

    .. code:: python

        from lifelines import AalenJohansenFitter
        from lifelines.datasets import load_waltons
        T, E = load_waltons()['T'], load_waltons()['E']
        ajf = AalenJohansenFitter(calculate_variance=True)
        ajf.fit(T, E, event_of_interest=1)
        ajf.cumulative_density_
        ajf.plot()


    References
    ----------
    If you are interested in learning more, we recommend the following open-access
    paper; Edwards JK, Hester LL, Gokhale M, Lesko CR. Methodologic Issues When Estimating Risks in
    Pharmacoepidemiology. Curr Epidemiol Rep. 2016;3(4):285-296.
    """

    def __init__(self, jitter_level=0.0001, seed=None, alpha=0.05, calculate_variance=True, **kwargs):
        NonParametricUnivariateFitter.__init__(self, alpha=alpha, **kwargs)
        self._jitter_level = jitter_level
        self._seed = seed  # Seed is for the jittering process
        self._calc_var = calculate_variance  # Optionally skips calculating variance to save time on bootstraps

    @CensoringType.right_censoring
    def fit(
        self,
        durations,
        event_observed,
        event_of_interest,
        timeline=None,
        entry=None,
        label=None,
        alpha=None,
        ci_labels=None,
        weights=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """
        Parameters
        ----------
          durations: an array or pd.Series of length n -- duration of subject was observed for
          event_observed: an array, or pd.Series, of length n. Integer indicator of distinct events. Must be
             only positive integers, where 0 indicates censoring.
          event_of_interest: integer -- indicator for event of interest. All other integers are considered competing events
             Ex) event_observed contains 0, 1, 2 where 0:censored, 1:lung cancer, and 2:death. If event_of_interest=1, then death (2)
             is considered a competing event. The returned cumulative incidence function corresponds to risk of lung cancer
          timeline: return the best estimate at the values in timelines (positively increasing)
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated (not left-censored) observations. If None, all members of the population
             were born at time 0.
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<1-alpha/2>
          weights: n array, or pd.Series, of length n, if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns
        -------
        self : AalenJohansenFitter
          self, with new properties like ``cumulative_incidence_``.
        """
        # Checking for tied event times
        ties = self._check_for_duplicates(durations=durations, events=event_observed)

        if ties:
            warnings.warn(
                dedent(
                    """Tied event times were detected. The Aalen-Johansen estimator cannot handle tied event times.
                To resolve ties, data is randomly jittered."""
                ),
                Warning,
            )
            durations = self._jitter(
                durations=pd.Series(durations), event=pd.Series(event_observed), jitter_level=self._jitter_level, seed=self._seed
            )

        alpha = alpha if alpha else self.alpha

        # Creating label for event of interest & indicator for that event
        event_of_interest = int(event_of_interest)
        cmprisk_label = "CIF_" + str(event_of_interest)
        self.label_cmprisk = "observed_" + str(event_of_interest)

        # Fitting Kaplan-Meier for either event of interest OR competing risk
        km = KaplanMeierFitter().fit(durations, event_observed=event_observed, timeline=timeline, entry=entry, weights=weights)
        aj = km.event_table
        aj["overall_survival"] = km.survival_function_
        aj["lagged_overall_survival"] = aj["overall_survival"].shift()

        # Setting up table for calculations and to return to user
        event_spec = pd.Series(event_observed) == event_of_interest
        self.durations, self.event_observed, *_, event_table, weights = _preprocess_inputs(
            durations=durations, event_observed=event_spec, timeline=timeline, entry=entry, weights=weights
        )
        event_spec_times = event_table["observed"]
        event_spec_times = event_spec_times.rename(self.label_cmprisk)
        aj = pd.concat([aj, event_spec_times], axis=1).reset_index()

        # Estimator of Cumulative Incidence (Density) Function
        aj[cmprisk_label] = (aj[self.label_cmprisk] / aj["at_risk"] * aj["lagged_overall_survival"]).cumsum()
        aj.loc[0, cmprisk_label] = 0  # Setting initial CIF to be zero
        aj = aj.set_index("event_at")

        # Setting attributes
        self._estimation_method = "cumulative_density_"
        self._estimate_name = "cumulative_density_"
        self.timeline = km.timeline

        self._label = coalesce(label, self._label, "AJ_estimate")
        self.cumulative_density_ = pd.DataFrame(aj[cmprisk_label])

        # Technically, cumulative incidence, but consistent with KaplanMeierFitter
        self.event_table = aj[["removed", "observed", self.label_cmprisk, "censored", "entrance", "at_risk"]]  # Event table

        if self._calc_var:
            self.variance_, self.confidence_interval_ = self._bounds(
                aj["lagged_overall_survival"], alpha=alpha, ci_labels=ci_labels
            )
        else:
            self.variance_, self.confidence_interval_ = None, None

        self.confidence_interval_cumulative_density_ = self.confidence_interval_
        return self

    def _jitter(self, durations, event, jitter_level, seed=None):
        """Determine extent to jitter tied event times. Automatically called by fit if tied event times are detected
        """
        np.random.seed(seed)

        if jitter_level <= 0:
            raise ValueError("The jitter level is less than zero, please select a jitter value greater than 0")

        event_times = durations[event != 0].copy()
        n = event_times.shape[0]

        # Determining extent to jitter event times up or down
        shift = np.random.uniform(low=-1, high=1, size=n) * jitter_level
        event_times += shift
        durations_jitter = event_times.align(durations)[0].fillna(durations)

        # Recursive call if event times are still tied after jitter
        if self._check_for_duplicates(durations=durations_jitter, events=event):
            return self._jitter(durations=durations_jitter, event=event, jitter_level=jitter_level, seed=seed)
        return durations_jitter

    def _bounds(self, lagged_survival, alpha, ci_labels):
        """Bounds are based on pg 411 of "Modelling Survival Data in Medical Research" David Collett 3rd Edition, which
        is derived from Greenwood's variance estimator. Confidence intervals are obtained using the delta method
        transformation of SE(log(-log(F_j))). This ensures that the confidence intervals all lie between 0 and 1.

        Formula for the variance follows:

        .. math::

            Var(F_j) = sum((F_j(t) - F_j(t_i))**2 * d/(n*(n-d) + S(t_i-1)**2 * ((d*(n-d))/n**3) +
                        -2 * sum((F_j(t) - F_j(t_i)) * S(t_i-1) * (d/n**2)

        Delta method transformation:

        .. math::

            SE(log(-log(F_j) = SE(F_j) / (F_j * |log(F_j)|)

        More information can be found at: https://support.sas.com/documentation/onlinedoc/stat/141/lifetest.pdf
        There is also an alternative method (Aalen) but this is not currently implemented
        """
        # Preparing environment
        ci = 1 - alpha
        df = self.event_table.copy()
        df["Ft"] = self.cumulative_density_
        df["lagS"] = lagged_survival.fillna(1)
        if ci_labels is None:
            ci_labels = ["%s_upper_%g" % (self._label, ci), "%s_lower_%g" % (self._label, ci)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        # Have to loop through each time independently. Don't think there is a faster way
        all_vars = []
        for _, r in df.iterrows():
            sf = df.loc[df.index <= r.name].copy()
            F_t = float(r["Ft"])
            first_term = np.sum((F_t - sf["Ft"]) ** 2 * sf["observed"] / sf["at_risk"] / (sf["at_risk"] - sf["observed"]))
            second_term = np.sum(
                sf["lagS"] ** 2
                / sf["at_risk"]
                * sf[self.label_cmprisk]
                / sf["at_risk"]
                * (sf["at_risk"] - sf[self.label_cmprisk])
                / sf["at_risk"]
            )
            third_term = np.sum((F_t - sf["Ft"]) / sf["at_risk"] * sf["lagS"] * sf[self.label_cmprisk] / sf["at_risk"])
            variance = first_term + second_term - 2 * third_term
            all_vars.append(variance)
        df["variance"] = all_vars

        # Calculating Confidence Intervals
        df["F_transformed"] = np.log(-np.log(df["Ft"]))
        df["se_transformed"] = np.sqrt(df["variance"]) / (df["Ft"] * np.absolute(np.log(df["Ft"])))
        zalpha = inv_normal_cdf(1 - alpha / 2)
        df[ci_labels[0]] = np.exp(-np.exp(df["F_transformed"] + zalpha * df["se_transformed"]))
        df[ci_labels[1]] = np.exp(-np.exp(df["F_transformed"] - zalpha * df["se_transformed"]))
        return df["variance"], df[ci_labels]

    @staticmethod
    def _check_for_duplicates(durations, events):
        """Checks for duplicated event times in the data set. This is narrowed to detecting duplicated event times
        where the events are of different types
        """
        # Setting up DataFrame to detect duplicates
        df = pd.DataFrame({"t": durations, "e": events})

        # Finding duplicated event times
        dup_times = df.loc[df["e"] != 0, "t"].duplicated(keep=False)

        # Finding duplicated events and event times
        dup_events = df.loc[df["e"] != 0, ["t", "e"]].duplicated(keep=False)

        # Detect duplicated times with different event types
        return (dup_times & (~dup_events)).any()

    def plot_cumulative_density(self, **kwargs):
        """Plots a pretty figure of the model

        Matplotlib plot arguments can be passed in inside the kwargs.

        Parameters
        -----------
        show_censors: bool
            place markers at censorship events. Default: False
        censor_styles: dict
            If show_censors, this dictionary will be passed into the plot call.
        ci_alpha: float
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
        if not self._calc_var:
            kwargs["ci_show"] = False
        _plot_estimate(self, estimate=self._estimate_name, **kwargs)
