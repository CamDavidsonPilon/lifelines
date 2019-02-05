# -*- coding: utf-8 -*-
from __future__ import print_function, division
import autograd.numpy as np
from autograd import hessian, value_and_grad
from scipy.optimize import minimize

from scipy import stats
from numpy.linalg import inv
from lifelines.fitters import ParametericUnivariateFitter
from lifelines.utils import (
    _to_array,
    inv_normal_cdf,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
)


class LogLogisticFitter(ParametericUnivariateFitter):

    r"""
    This class implements a Log-Logistic model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = (1 + (t/\alpha)^{\beta})^{-1},   \alpha > 0, \beta > 0,

    and the hazard rate is:

    .. math::  h(t) = (\beta/\alpha)(t / \alpha) ^ {\beta-1} / (1 + (t/\alpha)^{\beta})

    After calling the `.fit` method, you have access to properties like:
    ``cumulative_hazard_``, ``plot``, ``survival_function_``, ``alpha_`` and ``beta_``.

    A summary of the fit is available with the method 'print_summary()'
    
    Examples
    --------

    >>> from lifelines import LogLogisticFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> llf = WeibullFitter()
    >>> llf.fit(waltons['T'], waltons['E'])
    >>> llf.plot()
    >>> print(llf.alpha_)

    """
    _fitted_parameter_names = ["alpha_", "beta_"]

    @property
    def median_(self):
        return self.alpha_

    def _cumulative_hazard(self, params, times):
        alpha_, beta_ = params
        return np.log((times / alpha_) ** beta_ + 1)
