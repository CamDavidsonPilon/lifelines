# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import warnings

from lifelines.fitters import UnivariateFitter
from lifelines.utils import _preprocess_inputs, inv_normal_cdf
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter


class AalenJohansenFitter(UnivariateFitter):
    """Class for fitting the Aalen-Johansen estimate for the cumulative incidence function in a competing risks framework.
    Treating competing risks as censoring can result in over-estimated cumulative density functions. Using the Kaplan
    Meier estimator with competing risks as censored is akin to estimating the cumulative density if all competing risks
    had been prevented. If you are interested in learning more, I (Paul Zivich) recommend the following open-access
    paper; Edwards JK, Hester LL, Gokhale M, Lesko CR. Methodologic Issues When Estimating Risks in
    Pharmacoepidemiology. Curr Epidemiol Rep. 2016;3(4):285-296.

    AalenJohansenFitter(alpha=0.95, jitter_level=0.00001, seed=None)

    Aalen-Johansen cannot deal with tied times. We can get around this by randomy jittering the event times
    slightly. This will be done automatically and generates a warning.
    """

    def __init__(self, jitter_level=0.0001, seed=None, alpha=0.95):
        UnivariateFitter.__init__(self, alpha=alpha)
        self._jitter_level = jitter_level
        self._seed = seed  # Seed is for the jittering process

    def fit(
        self,
        durations,
        event_observed,
        event_of_interest,
        timeline=None,
        entry=None,
        label="AJ_estimate",
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
          timeline: return the best estimate at the values in timelines (postively increasing)
          entry: an array, or pd.Series, of length n -- relative time when a subject entered the study. This is
             useful for left-truncated (not left-censored) observations. If None, all members of the population
             were born at time 0.
          label: a string to name the column of the estimate.
          alpha: the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
          weights: n array, or pd.Series, of length n, if providing a weighted dataset. For example, instead
              of providing every subject as a single element of `durations` and `event_observed`, one could
              weigh subject differently.

        Returns
        -------
        self : AalenJohansenFitter
          self, with new properties like 'cumulative_incidence_'.
        """
        # Checking for tied event times
        if np.sum(pd.Series(durations).duplicated()) > 0:
            # Seeing if there is a large amount of ties in the data (>20%)
            if np.sum(pd.Series(durations).duplicated()) / len(durations) > 0.2:
                warnings.warn(
                    """It looks like there are many tied events in your data set. The Aalen-Johansen
                              estimator should only be used when there are no/few tied events""",
                    Warning,
                )
                # I am unaware of a recommended cut-off, but 20% would be suggestive of issues
            # Raise warning if duplicated times, then randomly jitter times
            warnings.warn(
                """Tied event times were detected. The Aalen-Johansen estimator cannot handle tied event times.
                To resolve ties, data is randomly jittered.""",
                Warning,
            )
            durations = self._jitter(
                durations=pd.Series(durations),
                event=pd.Series(event_observed),
                jitter_level=self._jitter_level,
                seed=self._seed,
            )

        # Creating label for event of interest & indicator for that event
        cmprisk_label = "CIF_" + str(int(event_of_interest))
        self.label_cmprisk = "observed_" + str(int(event_of_interest))

        # Fitting Kaplan-Meier for either event of interest OR competing risk
        km = KaplanMeierFitter()
        km.fit(durations, event_observed=event_observed, timeline=timeline, entry=entry, weights=weights)
        aj = km.event_table
        aj["overall_survival"] = km.survival_function_
        aj["lagged_overall_survival"] = aj["overall_survival"].shift()

        # Setting up table for calculations and to return to user
        event_spec = np.where(pd.Series(event_observed) == event_of_interest, 1, 0)
        event_spec_proc = _preprocess_inputs(
            durations=durations, event_observed=event_spec, timeline=timeline, entry=entry, weights=weights
        )
        event_spec_times = event_spec_proc[-1]["observed"]
        event_spec_times = event_spec_times.rename(self.label_cmprisk)
        aj = pd.concat([aj, event_spec_times], axis=1).reset_index()

        # Estimator of Cumulative Incidence (Density) Function
        aj[cmprisk_label] = ((aj[self.label_cmprisk]) / (aj["at_risk"]) * aj["lagged_overall_survival"]).cumsum()
        aj.loc[0, cmprisk_label] = 0  # Setting initial CIF to be zero
        aj = aj.set_index("event_at")

        # Setting attributes
        self._estimation_method = "cumulative_density_"
        self._estimate_name = "cumulative_density_"
        self._predict_label = label
        self._update_docstrings()

        alpha = alpha if alpha else self.alpha
        self._label = label
        self.cumulative_density_ = pd.DataFrame(aj[cmprisk_label])
        # Technically, cumulative incidence, but consistent with KaplanMeierFitter
        self.event_table = aj[
            ["removed", "observed", self.label_cmprisk, "censored", "entrance", "at_risk"]
        ]  # Event table
        self.variance, self.confidence_interval_ = self._bounds(
            aj["lagged_overall_survival"], alpha=alpha, ci_labels=ci_labels
        )
        return self

    def _jitter(self, durations, event, jitter_level, seed=None):
        """Determine extent to jitter tied event times. Automatically called by fit if tied event times are detected
        """
        if jitter_level <= 0:
            raise ValueError("The jitter level is less than zero, please select a jitter value greater than 0")
        if seed is not None:
            np.random.seed(seed)

        event_time = durations.loc[event != 0].copy()
        # Determining whether to randomly shift event times up or down
        mark = np.random.choice([-1, 1], size=event_time.shape[0])
        # Determining extent to jitter event times up or down
        shift = np.random.uniform(size=event_time.shape[0]) * jitter_level
        # Jittering times
        event_time += mark * shift
        durations_jitter = event_time.align(durations)[0].fillna(durations)

        # Recursive call if event times are still tied after jitter
        if np.sum(event_time.duplicated()) > 0:
            return self._jitter(durations=durations_jitter, event=event, jitter_level=jitter_level, seed=seed)
        return durations_jitter

    def _bounds(self, lagged_survival, alpha, ci_labels):
        """Bounds are based on pg411 of "Modelling Survival Data in Medical Research" David Collett 3rd Edition, which
        is derived from Greenwood's variance estimator. Confidence intervals are obtained using the delta method
        transformation of SE(log(-log(F_j))). This ensures that the confidence intervals all lie between 0 and 1.

        Formula for the variance follows:
        Var(F_j) = sum((F_j(t) - F_j(t_i))**2 * d/(n*(n-d) + S(t_i-1)**2 * ((d*(n-d))/n**3) +
                    -2 * sum((F_j(t) - F_j(t_i)) * S(t_i-1) * (d/n**2)

        Delta method transformation:
        SE(log(-log(F_j) = SE(F_j) / (F_j * absolute(log(F_j)))

        More information can be found at: https://support.sas.com/documentation/onlinedoc/stat/141/lifetest.pdf
        There is also an alternative method (Aalen) but this is not currently implemented
        """
        # Preparing environment
        df = self.event_table.copy()
        df["Ft"] = self.cumulative_density_
        df["lagS"] = lagged_survival.fillna(1)
        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._predict_label, alpha), "%s_lower_%.2f" % (self._predict_label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        # Have to loop through each time independently. Don't think there is a faster way
        all_vars = []
        for _, r in df.iterrows():
            sf = df.loc[df.index <= r.name].copy()
            F_t = float(r["Ft"])
            sf["part1"] = ((F_t - sf["Ft"]) ** 2) * (
                sf["observed"] / (sf["at_risk"] * (sf["at_risk"] - sf["observed"]))
            )
            sf["part2"] = (
                ((sf["lagS"]) ** 2)
                * sf[self.label_cmprisk]
                * ((sf["at_risk"] - sf[self.label_cmprisk]))
                / (sf["at_risk"] ** 3)
            )
            sf["part3"] = (F_t - sf["Ft"]) * sf["lagS"] * (sf[self.label_cmprisk] / (sf["at_risk"] ** 2))
            variance = (np.sum(sf["part1"])) + (np.sum(sf["part2"])) - 2 * (np.sum(sf["part3"]))
            all_vars.append(variance)
        df["variance"] = all_vars

        # Calculating Confidence Intervals
        df["F_transformed"] = np.log(-np.log(df["Ft"]))
        df["se_transformed"] = np.sqrt(df["variance"]) / (df["Ft"] * np.absolute(np.log(df["Ft"])))
        zalpha = inv_normal_cdf((1.0 + alpha) / 2.0)
        df[ci_labels[0]] = np.exp(-np.exp(df["F_transformed"] + zalpha * df["se_transformed"]))
        df[ci_labels[1]] = np.exp(-np.exp(df["F_transformed"] - zalpha * df["se_transformed"]))
        return df["variance"], df[ci_labels]
