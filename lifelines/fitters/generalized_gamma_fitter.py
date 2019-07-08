# -*- coding: utf-8 -*-
import autograd.numpy as np
from scipy.special import gammainccinv
from lifelines.fitters import KnownModelParametericUnivariateFitter
from lifelines.utils.gamma import gammaincc, gammainc, gammaln


class GeneralizedGammaFitter(KnownModelParametericUnivariateFitter):

    r"""

    This class implements a Generalized Gamma model for univariate data. The model has parameterized
    form:

    The cumulative hazard rate is

    .. math:: H(t) = \left(\frac{t}{\lambda}\right)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \frac{\rho}{\lambda}\left(\frac{t}{\lambda}\right)^{\rho-1}

    After calling the `.fit` method, you have access to properties like: ``cumulative_hazard_``, ``survival_function_``, ``alpha_``, ``lambda_`` and ``rho_``.
    A summary of the fit is available with the method ``print_summary()``.

    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.


    Examples
    --------

    >>> from lifelines import GeneralizedGammaFitter
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> ggf = GeneralizedGammaFitter()
    >>> ggf.fit(waltons['T'], waltons['E'])
    >>> ggf.plot()
    >>> ggf.summary

    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    cumumlative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    median_: float
        The median time to event
    lambda_: float
        The fitted parameter in the model
    rho_: float
        The fitted parameter in the model
    alpha_: float
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

    _fitted_parameter_names = ["alpha_", "lambda_", "rho_"]
    _bounds = [(0.0, None), (0.0, None), (0.0, None)]
    _initial_point = np.array([1.0, 1.0, 1.0])

    def _survival_function(self, params, times):
        alpha_, lambda_, rho_ = params
        ug = gammaincc(alpha_ / rho_, (times / lambda_) ** rho_)
        ug = np.clip(ug, 1e-17, 1 - 1e-17)
        return ug

    def _cumulative_hazard(self, params, times):
        sf = self._survival_function(params, times)
        return -np.log(sf)

    def _log_1m_sf(self, params, times):
        sf = self._survival_function(params, times)
        return np.log1p(-sf)

    def percentile(self, p):
        return self.lambda_ * gammainccinv(self.alpha_ / self.rho_, p) ** (1 / self.rho_)
