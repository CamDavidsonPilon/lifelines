# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd
import warnings
import autograd.numpy as np
from autograd import hessian, value_and_grad
from autograd.scipy.stats import norm 
from numpy.linalg import inv
from scipy.optimize import minimize

from lifelines.fitters import ParametericUnivariateFitter
from lifelines.utils import (
    _to_array,
    check_nans_or_infs,
    string_justify,
    format_floats,
    ConvergenceError,
    inv_normal_cdf,
    format_p_value,
)



class LogNormalFitter(ParametericUnivariateFitter):
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
        self,
        durations,
        event_observed=None,
        timeline=None,
        label="LogNormal_estimate",
        alpha=0.95,
        ci_labels=None,
        show_progress=False,
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
        self : LogNormalFitter
          self, with new properties like 'survival_function_', 'sigma_' and 'mu_'.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )

        # check for negative or 0 durations - these are not allowed in a log-normal model.
        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label

        (self.mu_, self.sigma_), self._log_likelihood, self._hessian_ = self._fit_model(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self._fitted_parameters_ = np.array([self.mu_, self.sigma_])
        self._fitted_parameters_names = ['mu_', 'sigma_']
        self.variance_matrix_ = inv(self._hessian_)

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(name=self._label)
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(name=self._label)
        self.median_ = np.exp(self.mu_)

        self.confidence_interval_ = self._bounds(alpha, ci_labels)

        # estimation methods
        self._predict_label = label
        self._update_docstrings()

        return self

    def _cumulative_hazard(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        cdf = norm.cdf(Z, loc=0, scale=1)
        cdf = np.clip(cdf, 0., 1 - 1e-14)
        return -np.log(1 - cdf)

    def _hazard(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_

        pdf = norm.pdf(Z, loc=0, scale=1)
        pdf = np.clip(pdf, 1e-14, np.inf)

        return pdf / (self._survival_function(params, times) * sigma_ * times)


    def _fit_model(self, T, E, show_progress=True):

        initial_values = np.array([0, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = minimize(
                value_and_grad(self._negative_log_likelihood),  # pylint: disable=no-value-for-parameter
                initial_values,
                jac=True,
                method="L-BFGS-B",
                args=(T, E),
                bounds=((None, None), (0.00001, None)),  # to stay well away from 0.
                options={"disp": show_progress},
            )

            if results.success:
                hessian_ = hessian(self._negative_log_likelihood)(results.x, T, E)  # pylint: disable=no-value-for-parameter
                return results.x, -results.fun, hessian_ * T.shape[0]
            print(results)
            raise ConvergenceError("Did not converge. This is a lifelines problem, not yours;")

    def _compute_z_values(self):
        return (self._fitted_parameters_ - np.array([0, 1])) / self._compute_standard_errors().loc["se"]

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
        print("{} = {}".format(justify("hypothesis"), 'mu_ != 0, sigma_ != 1'))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))

    def _bounds(self, alpha, ci_labels):
        """
        Necesary because of strange problem in 

        > make_jvp(norm.cdf)([1])(np.array([0,0]))

        """
        from scipy.stats import norm
        import numpy as np

        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)

        def _d_cumulative_hazard_d_mu(mu_, sigma_, log_T):
            Z = (log_T - mu_) / sigma_
            return -norm.pdf(Z) / norm.sf(Z) / sigma_

        def _d_cumulative_hazard_d_sigma(mu_, sigma_, log_T):
            Z = (log_T - mu_) / sigma_
            return -Z * norm.pdf(Z) / norm.sf(Z) / sigma_

        gradient_at_mle = np.stack(
            [
                _d_cumulative_hazard_d_mu(self.mu_, self.sigma_, np.log(self.timeline)),
                _d_cumulative_hazard_d_sigma(self.mu_, self.sigma_, np.log(self.timeline)),
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
