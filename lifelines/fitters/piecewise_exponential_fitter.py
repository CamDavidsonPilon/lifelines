# -*- coding: utf-8 -*-
import warnings
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

    Parameters
    -----------
    breakpoints: list
        a list of times when a new exponential model is constructed.
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Important
    ----------
    The parameterization of this model changed in lifelines 0.19.1. Previously, the cumulative hazard looked like
    :math:`\lambda_i t`. The parameterization is now the reciprocal of :math:`\lambda_i`.

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
    lambda_i_: float
        The fitted parameter in the model, for i = 0, 1 ... n-1 breakpoints
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    breakpoints: array
        The provided breakpoints

    """

    def __init__(self, breakpoints, *args, **kwargs):
        if (breakpoints is None) or (not list(breakpoints)):
            raise ValueError("Breakpoints must be provided.")

        if not (max(breakpoints) < np.inf):
            raise ValueError("Do not add inf to the breakpoints.")

        if min(breakpoints) <= 0:
            raise ValueError("First breakpoint must be greater than 0.")

        breakpoints = np.sort(breakpoints)
        self.breakpoints = np.append(breakpoints, [np.inf])
        n_breakpoints = len(self.breakpoints)

        self._fitted_parameter_names = ["lambda_%d_" % i for i in range(n_breakpoints)]

        super(PiecewiseExponentialFitter, self).__init__(*args, **kwargs)

    def _cumulative_hazard(self, params, times):
        warnings.simplefilter(action="ignore", category=FutureWarning)

        n = times.shape[0]
        times = times.reshape((n, 1))
        bp = self.breakpoints
        M = np.minimum(np.tile(bp, (n, 1)), times)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])
        return np.dot(M, 1 / params)
