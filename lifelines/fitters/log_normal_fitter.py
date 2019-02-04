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

    _fitted_parameter_names = ["mu_", "sigma_"]
    _bounds = [(None, None), (ParametericUnivariateFitter._MIN_PARAMETER_VALUE, None)]

    @property
    def median_(self):
        return np.exp(self.mu_)


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


    def _compute_confidence_bounds_of_cumulative_hazard(self, alpha, ci_labels):
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
