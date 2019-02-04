# -*- coding: utf-8 -*-
from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import autograd.numpy as autograd_np
from autograd import elementwise_grad as egrad

from scipy import stats
from numpy.linalg import inv
from lifelines.fitters import ParametericUnivariateFitter
from lifelines.utils import (
    _to_array,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
)


class WeibullFitter(ParametericUnivariateFitter):

    r"""
    This class implements a Weibull model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = exp(-(\lambda t)^\rho),   \lambda > 0, \rho > 0,

    which implies the cumulative hazard rate is

    .. math:: H(t) = (\lambda t)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \rho \lambda(\lambda t)^{\rho-1}

    After calling the `.fit` method, you have access to properties like:
    ``cumulative_hazard_``, ``survival_function_``, ``lambda_`` and ``rho_``.

    A summary of the fit is available with the method 'print_summary()'
    
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
        return (lambda_ * times) ** rho_

    @property
    def median_(self):
        return 1.0 / self.lambda_ * (np.log(2)) ** (1.0 / self.rho_)


