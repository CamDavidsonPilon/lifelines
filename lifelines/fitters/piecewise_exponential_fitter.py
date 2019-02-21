# -*- coding: utf-8 -*-
from __future__ import print_function, division
import autograd.numpy as np

from lifelines.fitters import KnownModelParametericUnivariateFitter


class PiecewiseExponentialFitter(KnownModelParametericUnivariateFitter):
    r"""
    This class implements an Piecewise Exponential model for univariate data. The model has parameterized
    hazard rate:

    .. math::  h(t) = \begin{cases}
                        1/\lambda_0,  & \text{if $t \le \tau_0$} \\
                        1/\lambda_1 & \text{if $\tau_0 < t \le \tau_1$} \\
                        1/\lambda_2 & \text{if $\tau_1 < t \le \tau_2$} \\
                        ...
                      \end{cases}

    You specify the breakpoints, :math:`\tau_i`, and *lifelines* will find the
    optional values for the parameters.

    After calling the `.fit` method, you have access to properties like: ``survival_function_``, ``plot``, ``cumulative_hazard_``
    A summary of the fit is available with the method ``print_summary()``

    Important
    ----------
    The parameterization of this model changed in lifelines 0.19.1. Previously, the cumulative hazard looked like
    :math:`\lambda_i t`. The parameterization is now the recipricol of :math:`\lambda_i`.


    """

    def __init__(self, breakpoints, *args, **kwargs):
        breakpoints = np.sort(breakpoints)
        if not (breakpoints[-1] < np.inf):
            raise ValueError("Do not add inf to the breakpoints.")

        if breakpoints[0] < 0:
            raise ValueError("First breakpoint must be greater than 0.")

        self.breakpoints = np.append(breakpoints, [np.inf])
        n_breakpoints = len(self.breakpoints)

        self._fitted_parameter_names = ["lambda_%d_" % i for i in range(n_breakpoints)]

        super(PiecewiseExponentialFitter, self).__init__(*args, **kwargs)

    def _cumulative_hazard(self, params, times):

        n = times.shape[0]
        times = times.reshape((n, 1))

        bp = self.breakpoints
        M = np.minimum(np.tile(bp, (n, 1)), times)
        M = np.hstack([M[:, (0,)], np.diff(M, axis=1)])
        return np.dot(M, 1 / params)
