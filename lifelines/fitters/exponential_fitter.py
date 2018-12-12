# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats

from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    inv_normal_cdf,
    check_nans_or_infs,
    significance_code,
    string_justify,
    significance_codes_as_text,
)


class ExponentialFitter(UnivariateFitter):
    r"""
    This class implements an Exponential model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = exp(-(\lambda*t)),   \lambda >0

    which implies the cumulative hazard rate is

    .. math::  H(t) = \lambda*t

    and the hazard rate is:

    .. math::  h(t) = \lambda

    After calling the `.fit` method, you have access to properties like:
     'survival_function_', 'lambda_'

    A summary of the fit is available with the method 'print_summary()'

    Notes
    -----
    Reference: https://www4.stat.ncsu.edu/~dzhang2/st745/chap3.pdf

    """

    def fit(
        self, durations, event_observed=None, timeline=None, label="Exponential_estimate", alpha=None, ci_labels=None
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
          duration: iterable
            an array, or pd.Series, of length n -- duration subject was observed for
          event_observed: iterable, optional
            an array, list, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          timeline: iterable, optional
            return the best estimate at the values in timelines (postively increasing)
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: list, optional
            add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>

        Returns
        -------
        self : ExpontentialFitter
          self, with new properties like 'survival_function_' and 'lambda_'.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )
        self.timeline = (
            np.sort(np.asarray(timeline))
            if timeline is not None
            else np.arange(int(self.durations.min()), int(self.durations.max()) + 1)
        )
        self._label = label

        # estimation
        D = self.event_observed.sum()
        T = self.durations.sum()

        self.lambda_ = D / T
        self._lambda_variance_ = self.lambda_ / T
        self._log_likelihood = np.log(self.lambda_) * D - self.lambda_ * T
        self.survival_function_ = pd.DataFrame(
            np.exp(-self.lambda_ * self.timeline), columns=[self._label], index=self.timeline
        )
        self.confidence_interval_ = self._bounds(alpha if alpha else self.alpha, ci_labels)
        self.median_ = 1.0 / self.lambda_ * (np.log(2))

        # estimation methods
        self._estimate_name = "survival_function_"
        self._predict_label = label
        self._update_docstrings()

        # plotting
        self.plot_survival_function_ = self.plot

        return self

    def _estimation_method(self, t):
        return np.exp(-self.lambda_ * t)

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
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

    def _compute_standard_errors(self):
        n = self.durations.shape[0]
        var_lambda_ = self.lambda_ ** 2 / n
        return pd.DataFrame([[np.sqrt(var_lambda_)]], index=["se"], columns=["lambda_"])

    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().loc["se"]
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        return pd.DataFrame(
            [np.array([self.lambda_]) + alpha2 * se, np.array([self.lambda_]) - alpha2 * se],
            columns=["lambda_"],
            index=["upper-bound", "lower-bound"],
        )

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper"""
        lower_upper_bounds = self._compute_confidence_bounds_of_parameters()
        df = pd.DataFrame(index=["lambda_"])
        df["coef"] = [self.lambda_]
        df["se(coef)"] = self._compute_standard_errors().loc["se"]
        df["lower %.2f" % self.alpha] = lower_upper_bounds.loc["lower-bound"]
        df["upper %.2f" % self.alpha] = lower_upper_bounds.loc["upper-bound"]
        df["p"] = self._compute_p_values()
        return df

    def _compute_z_values(self):
        return self.lambda_ / self._compute_standard_errors().loc["se"]

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        # pylint: disable=unnecessary-lambda
        justify = string_justify(18)
        print(self)
        print("{} = {}".format(justify("number of subjects"), self.durations.shape[0]))
        print("{} = {}".format(justify("number of events"), np.where(self.event_observed)[0].shape[0]))
        print("{} = {:.3f}".format(justify("log-likelihood"), self._log_likelihood), end="\n\n")

        df = self.summary
        df[""] = [significance_code(p) for p in df["p"]]
        print(df.to_string(float_format=lambda f: "{:4.4f}".format(f)))
        print("---")
        print(significance_codes_as_text(), end="\n\n")
