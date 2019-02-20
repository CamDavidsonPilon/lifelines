# -*- coding: utf-8 -*-
from __future__ import print_function, division
import autograd.numpy as np

from lifelines.fitters import KnownModelParametericUnivariateFitter


class WeibullFitter(KnownModelParametericUnivariateFitter):

    r"""

    This class implements a Weibull model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = \exp\left(\left(\frac{-t}{\lambda}\right)^\rho\right),   \lambda > 0, \rho > 0,

    which implies the cumulative hazard rate is

    .. math:: H(t) = \left(\frac{t}{\lambda}\right)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \frac{\rho}{\lambda}(t/\lambda )^{\rho-1}

    After calling the `.fit` method, you have access to properties like: ``cumulative_hazard_``, ``survival_function_``, ``lambda_`` and ``rho_``.
    A summary of the fit is available with the method ``print_summary()``.


    Important
    ----------
    The parameterization of this model changed in lifelines 0.19.0. Previously, the cumulative hazard looked like
    :math:`(\lambda t)^\rho`. The parameterization is now the recipricol of :math:`\lambda`.
    
    Examples
    --------

    >>> from lifelines import WeibullFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> wbf = WeibullFitter()
    >>> wbf.fit(waltons['T'], waltons['E'])
    >>> wbf.plot()
    >>> print(wbf.lambda_)

    """

    _fitted_parameter_names = ["lambda_", "rho_"]

    def _cumulative_hazard(self, params, times):
        lambda_, rho_ = params
        return (times / lambda_) ** rho_

    @property
    def median_(self):
        return self.lambda_ * (np.log(2) ** (1.0 / self.rho_))
