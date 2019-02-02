# -*- coding: utf-8 -*-
from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import autograd.numpy as anp
from autograd import hessian, value_and_grad
from scipy.optimize import minimize

from scipy import stats
from numpy.linalg import inv
from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    _to_array,
    inv_normal_cdf,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
)


def _negative_log_likelihood(params, T, E):
    n = T.shape[0]
    alpha_, beta_ = params
    ll = (
        E
        * (
            anp.log(beta_)
            - anp.log(alpha_)
            + (beta_ - 1) * anp.log(T)
            - (beta_ - 1) * anp.log(alpha_)
            - anp.log1p((T / alpha_) ** beta_)
        )
    ).sum() + -1 * anp.log1p((T / alpha_) ** beta_).sum()
    return -ll / n


class LogLogisticFitter(UnivariateFitter):

    r"""
    This class implements a Log-Logistic model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = (1 + (t/\alpha)^{\beta})^{-1},   \alpha > 0, \beta > 0,

    and the hazard rate is:

    .. math::  h(t) = (\beta/\alpha)(t / \alpha) ^ {\beta-1} / (1 + (t/\alpha)^{\beta})

    After calling the `.fit` method, you have access to properties like:
    ``cumulative_hazard_``, ``plot``, ``survival_function_``, ``alpha_`` and ``beta_``.

    A summary of the fit is available with the method 'print_summary()'
    
    Examples
    --------

    >>> from lifelines import LogLogisticFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> llf = WeibullFitter()
    >>> llf.fit(waltons['T'], waltons['E'])
    >>> llf.plot()
    >>> print(llf.alpha_)

    """

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label="LogLogistic_estimate",
        alpha=None,
        ci_labels=None,
        show_progress=False,
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
        durations: an array, or pd.Series
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
          self : LogLogisticFitter
            self with new properties like ``cumulative_hazard_``, ``survival_function_``, ``alpha_``, and ``beta_``.

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
        (self.alpha_, self.beta_), self._log_likelihood, self._hessian_ = self._fit_model(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self.variance_matrix_ = inv(self._hessian_)

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)

        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(self._label)

        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(self._label)

        self.confidence_interval_ = self._bounds(alpha, ci_labels)
        self.median_ = self.alpha_

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
        a, b = self.alpha_, self.beta_

        return pd.Series((b / a) * (times / a) ** (b - 1) / (1 + (times / a) ** b), index=_to_array(times))

    def survival_function_at_times(self, times):
        return pd.Series(np.exp(-self.cumulative_hazard_at_times(times)), index=_to_array(times))

    def cumulative_hazard_at_times(self, times):
        alpha_, beta_ = self.alpha_, self.beta_
        return pd.Series(np.log((times / alpha_) ** beta_ + 1), index=_to_array(times))

    def _fit_model(self, T, E, show_progress=True):

        initial_values = np.ones(2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = minimize(
                value_and_grad(_negative_log_likelihood),  # pylint: disable=no-value-for-parameter
                initial_values,
                jac=True,
                method="L-BFGS-B",
                args=(T, E),
                bounds=((0.000001, None), (0.000001, None)),  # to stay well away from 0.
                options={"disp": show_progress},
            )

            if results.success:
                hessian_ = hessian(_negative_log_likelihood)(results.x, T, E)  # pylint: disable=no-value-for-parameter
                return results.x, -results.fun, hessian_ * T.shape[0]
            print(results)
            raise ConvergenceError("Did not converge. This is a lifelines problem, not yours;")

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)

        def _d_cumulative_hazard_d_alpha_(alpha_, beta_, T):
            return -beta_ / alpha_ / ((alpha_ / T) ** beta_ + 1)

        def _d_cumulative_hazard_d_beta_(alpha_, beta_, T):
            return np.log(T / alpha_) / ((alpha_ / T) ** beta_ + 1)

        gradient_at_mle = np.stack(
            [
                _d_cumulative_hazard_d_alpha_(self.alpha_, self.beta_, self.timeline),
                _d_cumulative_hazard_d_beta_(self.alpha_, self.beta_, self.timeline),
            ]
        ).T

        std_cumulative_hazard = np.sqrt(
            np.einsum("nj,jk,nk->n", gradient_at_mle, self.variance_matrix_, gradient_at_mle)
        )

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = self.cumulative_hazard_at_times(self.timeline) + alpha2 * std_cumulative_hazard
        df[ci_labels[1]] = self.cumulative_hazard_at_times(self.timeline) - alpha2 * std_cumulative_hazard
        return df

    def _compute_standard_errors(self):
        var_alpha_, var_beta_ = self.variance_matrix_.diagonal()
        return pd.DataFrame([[np.sqrt(var_alpha_), np.sqrt(var_beta_)]], index=["se"], columns=["alpha_", "beta_"])

    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().loc["se"]
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        return pd.DataFrame(
            [np.array([self.alpha_, self.beta_]) + alpha2 * se, np.array([self.alpha_, self.beta_]) - alpha2 * se],
            columns=["alpha_", "beta_"],
            index=["upper-bound", "lower-bound"],
        )

    def _compute_z_values(self):
        return np.asarray([self.alpha_ - 1, self.beta_ - 1]) / self._compute_standard_errors().loc["se"]

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
        df = pd.DataFrame(index=["alpha_", "beta_"])
        df["coef"] = [self.alpha_, self.beta_]
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
        print("{} = {}".format(justify("hypothesis"), "alpha != 1, beta != 1"))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))
