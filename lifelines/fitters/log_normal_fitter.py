# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import autograd.numpy as np
from autograd.scipy.stats import norm
from lifelines.fitters import KnownModelParametericUnivariateFitter
from lifelines.utils.logsf import logsf


class LogNormalFitter(KnownModelParametericUnivariateFitter):
    r"""
    This class implements an Log Normal model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = 1 - \Phi((\log(t) - \mu)/\sigma),   \sigma >0

    where :math:`\Phi` is the CDF of a standard normal random variable.
    This implies the cumulative hazard rate is

    .. math::  H(t) = -\log(1 - \Phi((\log(t) - \mu)/\sigma))

    After calling the `.fit` method, you have access to properties like: ``survival_function_``, ``mu_``, ``sigma_``.
    A summary of the fit is available with the method ``print_summary()``

    """

    _fitted_parameter_names = ["mu_", "sigma_"]
    _bounds = [(None, None), (0, None)]

    @property
    def median_(self):
        return np.exp(self.mu_)

    def _cumulative_hazard(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        return -logsf(Z)

    def _log_hazard(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        return norm.logpdf(Z, loc=0, scale=1) - np.log(sigma_) - np.log(times) - logsf(Z)
