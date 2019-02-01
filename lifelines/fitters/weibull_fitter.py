# -*- coding: utf-8 -*-
from __future__ import print_function, division
import warnings
import time
import numpy as np
import pandas as pd
import autograd.numpy as autograd_np
from autograd import hessian, jacobian
from scipy.optimize import minimize, check_grad

from scipy import stats
from numpy.linalg import solve, norm, inv
from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    _to_array,
    inv_normal_cdf,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    ConvergenceWarning,
    format_p_value,
    format_floats,
)



def _negative_log_likelihood(lambda_rho, T, E):
    if np.any(np.asarray(lambda_rho) < 0):
        return 10e9
    n = T.shape[0]
    lambda_, rho = lambda_rho
    neg_ll = -autograd_np.log(rho * lambda_) * E.sum() - (rho - 1) * (E * autograd_np.log(lambda_ * T)).sum() + ((lambda_ * T) ** rho).sum()
    return neg_ll / n


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
    ``cumulative_hazard_``, ``survival_function_``, ``lambda_`` and ``rho_``.

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
        show_progress=True,
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
          self : WeibullFitter
            self with new properties like ``cumulative_hazard_``, ``survival_function_``, ``lambda_``, and ``rho_``.

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
        (self.lambda_, self.rho_), self._log_likelihood, self._hessian_ = self._newton_rhaphson(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self.variance_matrix_ = inv(self._hessian_)

        print("here1")
        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)
        print("here2")

        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(self._label)
        print("here3")

        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(self._label)
        print("here4")

        self.confidence_interval_ = self._bounds(alpha, ci_labels)
        self.median_ = 1.0 / self.lambda_ * (np.log(2)) ** (1.0 / self.rho_)
        print("here5")

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
        return pd.Series(self.lambda_ * self.rho_ * (self.lambda_ * times) ** (self.rho_ - 1), index=_to_array(times))

    def survival_function_at_times(self, times):
        return pd.Series(np.exp(-self.cumulative_hazard_at_times(times)), index=_to_array(times))

    @staticmethod
    def _cumulative_hazard_(params, t):
        lambda_, rho_ = params
        return (lambda_ * t) ** rho_

    def cumulative_hazard_at_times(self, times):
        return pd.Series(self._cumulative_hazard_([self.lambda_, self.rho_], times), index=_to_array(times))


    def _newton_rhaphson(self, T, E, show_progress=True):
        from lifelines.utils import _smart_search
        
        initial_values = np.array([T.mean(), T.mean()])
        initial_values = _smart_search(_negative_log_likelihood, 2, T, E)
        print(initial_values)
        n = T.shape[0]

        jac = jacobian(_negative_log_likelihood)
        hess =  hessian(_negative_log_likelihood)
        
        results = minimize(
            _negative_log_likelihood,
            initial_values,
            jac=jac,
            hess= hess,
            method='Newton-CG',
            args=(T, E),
            options={'disp': show_progress}
        )  

        if results.success:
            hessian_ = hess(results.x, T, E)
            return results.x, -results.fun, hessian_ * n
        raise ConvergenceError("Did not converge. This is a lifelines problem, not yours;")


    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)
        import pdb
        pdb.set_trace()
        gradient_at_mle = jacobian(self._cumulative_hazard_)(np.array([self.lambda_, self.rho_]), self.timeline)

        print("here")
        std_cumulative_hazard = np.sqrt(np.einsum('nj,jk,nk->n', gradient_at_mle, self.variance_matrix_, gradient_at_mle))
        print("here")


        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = self.cumulative_hazard_at_times(self.timeline) + alpha2 * std_cumulative_hazard
        df[ci_labels[1]] = self.cumulative_hazard_at_times(self.timeline) - alpha2 * std_cumulative_hazard
        return df

    def _compute_standard_errors(self):
        var_lambda_, var_rho_ = self.variance_matrix_.diagonal()
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
        return np.asarray([self.lambda_ - 1, self.rho_ - 1]) / self._compute_standard_errors().loc["se"]

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
        print("{} = {}".format(justify("hypothesis"), "lambda != 1, rho != 1"))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))
