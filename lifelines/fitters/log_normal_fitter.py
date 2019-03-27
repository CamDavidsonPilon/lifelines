# -*- coding: utf-8 -*-


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

    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.


    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    confidence_interval_cumulative_hazard_ : DataFrame
        The lower and upper confidence intervals for the cumulative hazard
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    confidence_interval_hazard_ : DataFrame
        The lower and upper confidence intervals for the hazard
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    confidence_interval_survival_function_ : DataFrame
        The lower and upper confidence intervals for the survival function
    cumumlative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    confidence_interval_cumumlative_density_ : DataFrame
        The lower and upper confidence intervals for the cumulative density
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    median_: float
        The median time to event
    mu_: float
        The fitted parameter in the model
    sigma_: float
        The fitted parameter in the model
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
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

    def _log_1m_sf(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        return norm.logcdf(Z, loc=0, scale=1)
