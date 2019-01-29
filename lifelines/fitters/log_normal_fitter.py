# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
from numpy import log
from scipy.optimize import minimize
from scipy.integrate import cumtrapz


from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    _to_array,
    check_nans_or_infs,
    string_justify,
    format_floats,
    ConvergenceError,
    inv_normal_cdf,
    dataframe_interpolate_at_times,
    format_p_value,
)


def _negative_log_likelihood(params, log_T, E):
    mu, sigma = params
    if sigma <= 0.00001:
        return 1e10
    Z = (log_T - mu) / sigma
    log_sf = norm.logsf(Z)

    ll = (E * (norm.logpdf(Z) - log_T - log(sigma) - log_sf)).sum() + log_sf.sum()
    return -ll


def diff_norm_pdf(z):
    return -z * norm.pdf(z)


def _mu_gradient(params, log_T, E):
    mu, sigma = params

    Z = (log_T - mu) / sigma
    dZ_dmu = -1 / sigma

    sf = norm.sf(Z)

    return (
        -(E * (diff_norm_pdf(Z) * dZ_dmu / norm.pdf(Z) + norm.pdf(Z) * dZ_dmu / sf)).sum()
        + (norm.pdf(Z) / sf * dZ_dmu).sum()
    )


def _sigma_gradient(params, log_T, E):
    mu, sigma = params

    Z = (log_T - mu) / sigma
    dZ_dsigma = -(log_T - mu) / sigma ** 2

    sf = norm.sf(Z)

    return (
        -(E * (diff_norm_pdf(Z) * dZ_dsigma / norm.pdf(Z) - 1 / sigma + norm.pdf(Z) * dZ_dsigma / sf)).sum()
        + (norm.pdf(Z) / sf * dZ_dsigma).sum()
    )


class LogNormalFitter(UnivariateFitter):
    r"""
    This class implements an Log Normal model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = 1 - \Phi((log(t) - \mu)/\sigma),   \sigma >0

    where :math:`\Phi` is the CDF of a standard normal random variable. 
    This implies the cumulative hazard rate is

    .. math::  H(t) = -log(1 - \Phi((log(t) - \mu)/\sigma))


    After calling the `.fit` method, you have access to properties like:
     'survival_function_', 'mu_', 'sigma_'

    A summary of the fit is available with the method 'print_summary()'

    """

    def fit(
        self, durations, event_observed=None, timeline=None, label="LogNormal_estimate", alpha=0.95, ci_labels=None
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
          durations: iterable
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
          self, with new properties like 'survival_function_', 'sigma_' and 'mu_'.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )
        n = self.durations.shape[0]

        # check for negative or 0 durations - these are not allowed in a weibull model.
        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label

        (self.mu_, self.sigma_), self._log_likelihood, self.variance_matrix_ = self._fit_model(
            self.durations, self.event_observed
        )

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(name=self._label)
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(name=self._label)
        self.median_ = np.exp(self.mu_)
        # self.confidence_interval_ = self._bounds(alpha, ci_labels)

        # estimation methods
        self._estimate_name = "cumulative_hazard_"
        self._predict_label = label
        self._update_docstrings()

        # plotting - Cumulative hazard takes priority.
        self.plot_cumulative_hazard = self.plot

        return self

    def _estimation_method(self, t):
        return self.survival_function_at_times(t)

    def hazard_at_times(self, times):
        return pd.Series(
            norm.pdf((log(times) - self.mu_) / self.sigma_)
            / (self.sigma_ * times * self.survival_function_at_times(times)),
            index=_to_array(times),
        )

    def survival_function_at_times(self, times):
        """
        Return a Pandas series of the predicted survival value at specific times

        Parameters
        -----------
        times: iterable or float

        Returns
        --------
        pd.Series

        """
        return pd.Series(1 - norm.cdf((log(times) - self.mu_) / self.sigma_), index=_to_array(times))

    def cumulative_hazard_at_times(self, times):
        return pd.Series(-log(1 - norm.cdf((log(times) - self.mu_) / self.sigma_)), index=_to_array(times))

    def _fit_model(self, T, E, initial_values=None):
        if initial_values is None:
            initial_values = np.array([log(T).mean() + 1, log(T).std()])

        def gradient_function(parameters, log_T, E):
            return np.array([_mu_gradient(parameters, log_T, E), _sigma_gradient(parameters, log_T, E)])

        results = minimize(
            _negative_log_likelihood, initial_values, args=(log(T), E), jac=gradient_function, method="BFGS"
        )
        if results.success:
            return results.x, -results.fun, results.hess_inv
        else:
            raise ConvergenceError("Did not converge.")

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper
        """
        lower_upper_bounds = self._compute_confidence_bounds_of_parameters()
        df = pd.DataFrame(index=["mu_", "sigma_"])
        df["coef"] = [self.mu_, self.sigma_]
        df["se(coef)"] = self._compute_standard_errors().loc["se"]
        df["lower %.2f" % self.alpha] = lower_upper_bounds.loc["lower-bound"]
        df["upper %.2f" % self.alpha] = lower_upper_bounds.loc["upper-bound"]
        df["p"] = self._compute_p_values()
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore")
            df["-log2(p)"] = -np.log2(df["p"])
        return df

    def print_summary(self, decimals=2, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing 
            multiple outputs. 

        """
        justify = string_justify(18)
        print(self)
        print("{} = {}".format(justify("number of subjects"), self.durations.shape[0]))
        print("{} = {}".format(justify("number of events"), np.where(self.event_observed)[0].shape[0]))
        print("{} = {:.3f}".format(justify("log-likelihood"), self._log_likelihood))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print("hyp: mu != 0, sigma != 1")
        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))

    def _compute_standard_errors(self):
        var_mu_, var_sigma_ = self.variance_matrix_.diagonal()
        return pd.DataFrame([[np.sqrt(var_mu_), np.sqrt(var_sigma_)]], index=["se"], columns=["mu_", "sigma_"])

    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().loc["se"]
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        return pd.DataFrame(
            [np.array([self.mu_, self.sigma_]) + alpha2 * se, np.array([self.mu_, self.sigma_]) - alpha2 * se],
            columns=["mu_", "sigma_"],
            index=["upper-bound", "lower-bound"],
        )

    def _compute_z_values(self):
        # Note that we compare sigma (the scale parameter) to the standard value of 1.
        return np.asarray([self.mu_, self.sigma_ - 1]) / self._compute_standard_errors().loc["se"]

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)
        var_mu_, var_sigma_ = self.variance_matrix_.diagonal()

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = self.cumulative_hazard_at_times(self.timeline) + alpha2 * std_cumulative_hazard
        df[ci_labels[1]] = self.cumulative_hazard_at_times(self.timeline) - alpha2 * std_cumulative_hazard
        return df
