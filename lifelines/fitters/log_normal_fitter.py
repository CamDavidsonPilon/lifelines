# -*- coding: utf-8 -*-

from scipy.special import erfinv
import autograd.numpy as np
from autograd.scipy.stats import norm
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils import CensoringType


class LogNormalFitter(KnownModelParametricUnivariateFitter):
    r"""
    This class implements an Log Normal model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = 1 - \Phi\left(\frac{\log(t) - \mu}{\sigma}\right),  \;\; \sigma >0

    where :math:`\Phi` is the CDF of a standard normal random variable.
    This implies the cumulative hazard rate is

    .. math::  H(t) = -\log\left(1 - \Phi\left(\frac{\log(t) - \mu}{\sigma}\right)\right)

    For inference, our null hypothesis is that mu=0.0, and sigma=1.0.

    After calling the ``.fit`` method, you have access to properties like: ``survival_function_``, ``mu_``, ``sigma_``.
    A summary of the fit is available with the method ``print_summary()``

    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.


    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    cumulative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    density_: DataFrame
        The estimated density function (PDF) (with custom timeline if provided)

    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    median_survival_time_: float
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

    mu_: float
    sigma_: float
    _fitted_parameter_names = ["mu_", "sigma_"]
    _bounds = [(None, None), (0, None)]
    _compare_to_values = np.array([0.0, 1.0])

    def _create_initial_point(self, Ts, E, *args):
        if CensoringType.is_right_censoring(self):
            log_T = np.log(Ts[0])
        elif CensoringType.is_left_censoring(self):
            log_T = np.log(Ts[1])
        elif CensoringType.is_interval_censoring(self):
            if E.sum() > 0:
                log_T = np.log(Ts[1][E.astype(bool)])
            else:
                log_T = np.array([0])
        return np.array([np.median(log_T), 1.0])

    @property
    def median_survival_time_(self) -> float:
        return np.exp(self.mu_)

    def percentile(self, p) -> float:
        return np.exp(self.mu_ + np.sqrt(2 * self.sigma_ ** 2) * erfinv(1 - 2 * p))

    def _cumulative_hazard(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        return -norm.logsf(Z)

    def _log_hazard(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        return norm.logpdf(Z, loc=0, scale=1) - np.log(sigma_) - np.log(times) - norm.logsf(Z)

    def _log_1m_sf(self, params, times):
        mu_, sigma_ = params
        Z = (np.log(times) - mu_) / sigma_
        return norm.logcdf(Z)
