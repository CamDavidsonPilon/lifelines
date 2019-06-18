# -*- coding: utf-8 -*-


import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter


class ExponentialFitter(KnownModelParametericUnivariateFitter):
    r"""
    This class implements an Exponential model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = \exp\left(\frac{-t}{\lambda}\right),   \lambda >0

    which implies the cumulative hazard rate is

    .. math::  H(t) = \frac{t}{\lambda}

    and the hazard rate is:

    .. math::  h(t) = \frac{1}{\lambda}

    After calling the `.fit` method, you have access to properties like: ``survival_function_``, ``lambda_``, ``cumulative_hazard_``
    A summary of the fit is available with the method ``print_summary()``

    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Important
    ----------
    The parameterization of this model changed in lifelines 0.19.0. Previously, the cumulative hazard looked like
    :math:`\lambda t`. The parameterization is now the reciprocal of :math:`\lambda`.

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
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    median_: float
        The median time to event
    lambda_: float
        The fitted parameter in the model
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    cumumlative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    confidence_interval_cumumlative_density_ : DataFrame
        The lower and upper confidence intervals for the cumulative density
    """

    _fitted_parameter_names = ["lambda_"]

    @property
    def median_(self):
        return np.log(2) / self.lambda_

    def _cumulative_hazard(self, params, times):
        lambda_ = params[0]
        return times / lambda_
