# -*- coding: utf-8 -*-
from __future__ import print_function, division
import warnings
import time
import numpy as np
import pandas as pd

from scipy import stats
from numpy.linalg import solve, norm, inv
from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    inv_normal_cdf,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    significance_code,
    ConvergenceWarning,
    significance_codes_as_text,
)


def _negative_log_likelihood(lambda_rho, T, E):
    if np.any(np.asarray(lambda_rho) < 0):
        return 10e9
    lambda_, rho = lambda_rho
    return -np.log(rho * lambda_) * E.sum() - (rho - 1) * (E * np.log(lambda_ * T)).sum() + ((lambda_ * T) ** rho).sum()


def _lambda_gradient(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return -rho * (E / lambda_ - (lambda_ * T) ** rho / lambda_).sum()


def _rho_gradient(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return -E.sum() / rho - (np.log(lambda_ * T) * E).sum() + (np.log(lambda_ * T) * (lambda_ * T) ** rho).sum()


def _d_rho_d_rho(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (1.0 / rho ** 2 * E + (np.log(lambda_ * T) ** 2 * (lambda_ * T) ** rho)).sum()


def _d_lambda_d_lambda_(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (rho / lambda_ ** 2) * (E + (rho - 1) * (lambda_ * T) ** rho).sum()


def _d_rho_d_lambda_(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (-1.0 / lambda_) * (E - (lambda_ * T) ** rho - rho * (lambda_ * T) ** rho * np.log(lambda_ * T)).sum()


class WeibullFitter(UnivariateFitter):

    r"""
    This class implements a Weibull model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = exp(-(\lambda t)^\rho),   \lambda > 0, \rho > 0,

    which implies the cumulative hazard rate is

    .. math:: H(t) = (\lambda t)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \rho \lambda(\lambda t)^{\rho-1}

    After calling the `.fit` method, you have access to properties like:
    cumulative_hazard_, survival_function_, lambda_ and rho_.

    A summary of the fit is available with the method 'print_summary()'
    
    Examples
    --------

    >>> from lifelines import WeibullFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> wbf = WeibullFitter()
    >>> wbf.fit(waltons['T'], waltons['E'])
    >>> wbf.plot()
    >>> print(wbf.lambda_)

    """

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label="Weibull_estimate",
        alpha=None,
        ci_labels=None,
        show_progress=False,
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
        duration: an array, or pd.Series
          length n, duration subject was observed for
        event_observed: numpy array or pd.Series, optional
          length n, True if the the death was observed, False if the event
           was lost (right-censored). Defaults all True if event_observed==None
        timeline: list, optional
            return the estimate at the values in timeline (postively increasing)
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: list, optional
            add custom column names to the generated confidence intervals
              as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
        show_progress: boolean, optional
            since this is an iterative fitting algorithm, switching this to True will display some iteration details.

        Returns
        -------
          self : WeibullFitter
            self with new properties like cumulative_hazard_, survival_function_, lambda_, and rho_.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        # check for negative or 0 durations - these are not allowed in a weibull model.
        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label
        alpha = alpha if alpha is not None else self.alpha

        # estimation
        (self.lambda_, self.rho_), self._hessian_ = self._newton_rhaphson(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self._log_likelihood = -_negative_log_likelihood((self.lambda_, self.rho_), self.durations, self.event_observed)
        self.variance_matrix_ = -inv(self._hessian_)
        self.survival_function_ = pd.DataFrame(
            self.survival_function_at_times(self.timeline), columns=[self._label], index=self.timeline
        )
        self.hazard_ = pd.DataFrame(self.hazard_at_times(self.timeline), columns=[self._label], index=self.timeline)
        self.cumulative_hazard_ = pd.DataFrame(
            self.cumulative_hazard_at_times(self.timeline), columns=[self._label], index=self.timeline
        )
        self.confidence_interval_ = self._bounds(alpha, ci_labels)
        self.median_ = 1.0 / self.lambda_ * (np.log(2)) ** (1.0 / self.rho_)

        # estimation methods
        self._estimate_name = "cumulative_hazard_"
        self._predict_label = label
        self._update_docstrings()

        # plotting - Cumulative hazard takes priority.
        self.plot_cumulative_hazard = self.plot

        return self

    def _estimation_method(self, t):
        return np.exp(-(self.lambda_ * t) ** self.rho_)

    def hazard_at_times(self, times):
        return self.lambda_ * self.rho_ * (self.lambda_ * times) ** (self.rho_ - 1)

    def survival_function_at_times(self, times):
        return np.exp(-self.cumulative_hazard_at_times(times))

    def cumulative_hazard_at_times(self, times):
        return (self.lambda_ * times) ** self.rho_

    def _newton_rhaphson(self, T, E, precision=1e-5, show_progress=False):  # pylint: disable=too-many-locals
        from lifelines.utils import _smart_search

        def hessian_function(parameters, T, E):
            return np.array(
                [
                    [_d_lambda_d_lambda_(parameters, T, E), _d_rho_d_lambda_(parameters, T, E)],
                    [_d_rho_d_lambda_(parameters, T, E), _d_rho_d_rho(parameters, T, E)],
                ]
            )

        def gradient_function(parameters, T, E):
            return np.array([_lambda_gradient(parameters, T, E), _rho_gradient(parameters, T, E)])

        # initialize the parameters. This shows dramatic improvements.
        parameters = _smart_search(_negative_log_likelihood, 2, T, E)
        i = 1
        step_size = 0.9
        max_steps = 50
        converging, completed = True, False
        start = time.time()

        while converging and i < max_steps:
            # Do not override hessian and gradient in case of garbage
            h, g = (hessian_function(parameters, T, E), gradient_function(parameters, T, E))

            delta = solve(h, -step_size * g.T)
            if np.any(np.isnan(delta)):
                raise ConvergenceError("delta contains nan value(s). Convergence halted.")

            parameters += delta

            # Save these as pending result
            hessian = h

            if show_progress:
                print(
                    "Iteration %d: norm_delta = %.5f, seconds_since_start = %.1f"
                    % (i, norm(delta), time.time() - start)
                )

            if norm(delta) < precision:
                converging = False
                completed = True
            i += 1

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        if not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return parameters, hessian

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)
        var_lambda_, var_rho_ = inv(self._hessian_).diagonal()

        def _dH_d_lambda(lambda_, rho, T):
            return rho / lambda_ * (lambda_ * T) ** rho

        def _dH_d_rho(lambda_, rho, T):
            return np.log(lambda_ * T) * (lambda_ * T) ** rho

        def sensitivity_analysis(lambda_, rho, var_lambda_, var_rho_, T):
            return var_lambda_ * _dH_d_lambda(lambda_, rho, T) ** 2 + var_rho_ * _dH_d_rho(lambda_, rho, T) ** 2

        std_cumulative_hazard = np.sqrt(
            sensitivity_analysis(self.lambda_, self.rho_, var_lambda_, var_rho_, self.timeline)
        )

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = self.cumulative_hazard_at_times(self.timeline) + alpha2 * std_cumulative_hazard
        df[ci_labels[1]] = self.cumulative_hazard_at_times(self.timeline) - alpha2 * std_cumulative_hazard
        return df

    def _compute_standard_errors(self):
        var_lambda_, var_rho_ = inv(self._hessian_).diagonal()
        return pd.DataFrame([[np.sqrt(var_lambda_), np.sqrt(var_rho_)]], index=["se"], columns=["lambda_", "rho_"])

    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().loc["se"]
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        return pd.DataFrame(
            [np.array([self.lambda_, self.rho_]) + alpha2 * se, np.array([self.lambda_, self.rho_]) - alpha2 * se],
            columns=["lambda_", "rho_"],
            index=["upper-bound", "lower-bound"],
        )

    def _compute_z_values(self):
        return np.asarray([self.lambda_, self.rho_]) / self._compute_standard_errors().loc["se"]

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

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
        df = pd.DataFrame(index=["lambda_", "rho_"])
        df["coef"] = [self.lambda_, self.rho_]
        df["se(coef)"] = self._compute_standard_errors().loc["se"]
        df["lower %.2f" % self.alpha] = lower_upper_bounds.loc["lower-bound"]
        df["upper %.2f" % self.alpha] = lower_upper_bounds.loc["upper-bound"]
        df["p"] = self._compute_p_values()
        return df

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
