# -*- coding: utf-8 -*-
from __future__ import print_function, division
import autograd.numpy as np

from lifelines.fitters import ParametericUnivariateFitter


class PiecewiseExponentialFitter(ParametericUnivariateFitter):
    def __init__(self, breakpoints, *args, **kwargs):
        breakpoints = np.sort(breakpoints)
        if breakpoints[-1] < np.inf:
            raise ValueError("Do not add inf to the breakpoints.")

        if breakpoints[0] > 0:
            raise ValueError("First breakpoint must be greater than 0.")

        self.breakpoints = np.append(breakpoints, [np.inf])
        n_breakpoints = len(self.breakpoints)

        self._fitted_parameter_names = ["lambda_%d_" % i for i in range(n_breakpoints)]

        super(PieceWiseExponential, self).__init__(*args, **kwargs)

    def _cumulative_hazard(self, params, times):
        n = times.shape[0]
        times = times.reshape((n, 1))

        bp = self.breakpoints
        M = np.minimum(np.tile(bp, (n, 1)), times)
        M = np.hstack([M[:, [0]], np.diff(M, axis=1)])
        return np.dot(M, params)
