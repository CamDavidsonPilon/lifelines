# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter


class ExponentialFitter(KnownModelParametericUnivariateFitter):
    r"""
    This class implements an Exponential model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = \exp(-\lambda t),   \lambda >0

    which implies the cumulative hazard rate is

    .. math::  H(t) = \lambda t

    and the hazard rate is:

    .. math::  h(t) = \lambda

    After calling the `.fit` method, you have access to properties like: ``survival_function_``, ``lambda_``, ``cumulative_hazard_``
    A summary of the fit is available with the method ``print_summary()``

    Notes
    -----
    Reference: https://www4.stat.ncsu.edu/~dzhang2/st745/chap3.pdf

    """

    _fitted_parameter_names = ["lambda_"]

    @property
    def median_(self):
        return 1.0 / self.lambda_ * (np.log(2))

    def _fit_model(self, T, E, entry, show_progress=False):
        T = T - entry
        lambda_ = E.sum() / T.sum()
        lambda_variance_ = lambda_ / T.sum()
        log_likelihood = np.log(lambda_) * E.sum() - lambda_ * T.sum()
        return [lambda_], log_likelihood, np.array([[1.0 / lambda_variance_]])

    def _cumulative_hazard(self, params, times):
        lambda_ = params[0]
        return lambda_ * times
