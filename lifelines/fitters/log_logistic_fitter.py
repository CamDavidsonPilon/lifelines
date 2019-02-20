# -*- coding: utf-8 -*-
from __future__ import print_function, division
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

    Examples
    --------

    >>> from lifelines import LogLogisticFitter
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> llf = LogLogisticFitter()
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
        return np.log1p((times / alpha_) ** beta_)
