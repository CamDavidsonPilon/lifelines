# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
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

    Important
    ----------
    The parameterization of this model changed in lifelines 0.19.0. Previously, the cumulative hazard looked like
    :math:`\lambda t`. The parameterization is now the recipricol of :math:`\lambda`.


    Notes
    -----
    Reference: https://www4.stat.ncsu.edu/~dzhang2/st745/chap3.pdf

    """

    _fitted_parameter_names = ["lambda_"]

    @property
    def median_(self):
        return np.log(2) / self.lambda_

    def _fit_model(self, T, E, entry, show_progress=False):
        T = T - entry
        lambda_ = T.sum() / E.sum()
        lambda_variance_ = T.sum() / lambda_
        log_likelihood = -np.log(lambda_) * E.sum() + lambda_ * T.sum()
        return [lambda_], log_likelihood, np.array([[1.0 / lambda_variance_]])

    def _cumulative_hazard(self, params, times):
        lambda_ = params[0]
        return times / lambda_
