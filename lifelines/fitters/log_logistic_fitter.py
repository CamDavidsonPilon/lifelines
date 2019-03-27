# -*- coding: utf-8 -*-
import autograd.numpy as np

from lifelines.fitters import KnownModelParametericUnivariateFitter


class LogLogisticFitter(KnownModelParametericUnivariateFitter):

    r"""
    This class implements a Log-Logistic model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = \left(1 + \left(\frac{t}{\alpha}\right)^{\beta}\right)^{-1},   \alpha > 0, \beta > 0,

    and the hazard rate is:

    .. math::  h(t) = \frac{\left(\frac{\beta}{\alpha}\right)\left(\frac{t}{\alpha}\right) ^ {\beta-1}}{\left(1 + \left(\frac{t}{\alpha}\right)^{\beta}\right)}

    and the cumulative hazard is:

    .. math:: H(t) = \log\left(\left(\frac{t}{\alpha}\right) ^ {\beta} + 1\right)

    After calling the `.fit` method, you have access to properties like: ``cumulative_hazard_``, ``plot``, ``survival_function_``, ``alpha_`` and ``beta_``.
    A summary of the fit is available with the method 'print_summary()'

    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Examples
    --------

    >>> from lifelines import LogLogisticFitter
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> llf = LogLogisticFitter()
    >>> llf.fit(waltons['T'], waltons['E'])
    >>> llf.plot()
    >>> print(llf.alpha_)

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
    alpha_: float
        The fitted parameter in the model
    beta_: float
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
    _fitted_parameter_names = ["alpha_", "beta_"]

    @property
    def median_(self):
        return self.alpha_

    def _cumulative_hazard(self, params, times):
        alpha_, beta_ = params
        return np.log1p((times / alpha_) ** beta_)

    def _log_1m_sf(self, params, times):
        alpha_, beta_ = params
        return -np.log1p((times / alpha_) ** -beta_)
