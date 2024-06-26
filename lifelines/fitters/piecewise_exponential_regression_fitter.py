# -*- coding: utf-8 -*-
import autograd.numpy as np
import pandas as pd

from lifelines.utils import coalesce, _get_index, CensoringType
from lifelines.fitters import ParametricRegressionFitter
from lifelines.utils.safe_exp import safe_exp


class PiecewiseExponentialRegressionFitter(ParametricRegressionFitter):
    r"""
    This implements a piecewise constant-hazard model at pre-specified break points.


    .. math::  h(t) = \begin{cases}
                        1/\lambda_0(x)  & \text{if $t \le \tau_0$} \\
                        1/\lambda_1(x) & \text{if $\tau_0 < t \le \tau_1$} \\
                        1/\lambda_2(x) & \text{if $\tau_1 < t \le \tau_2$} \\
                        ...
                      \end{cases}

    where :math:`\lambda_i(x) = \exp{\beta_i x}`.

    Parameters
    -----------
    breakpoints: list
        a list of times when a new exponential model is constructed.
    penalizer: float
        penalize the variance of the :math:`\lambda_i`. See blog post below.
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Examples
    ----------
    See blog post `here <https://dataorigami.net/blogs/napkin-folding/churn>`_ and
    paper replication `here <https://github.com/CamDavidsonPilon/lifelines-replications/blob/master/replications/Friedman_1982.ipynb>`_

    """

    _FAST_MEDIAN_PREDICT = True  # mmm not really...

    # about 50% faster than BFGS
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-6, "maxiter": 200}
    fit_intercept = True

    def __init__(self, breakpoints, alpha=0.05, penalizer=0.0):
        super(PiecewiseExponentialRegressionFitter, self).__init__(alpha=alpha)

        breakpoints = np.sort(breakpoints)
        if len(breakpoints) and not (breakpoints[-1] < np.inf):
            raise ValueError("Do not add inf to the breakpoints.")

        if len(breakpoints) and breakpoints[0] < 0:
            raise ValueError("First breakpoint must be greater than 0.")

        self.breakpoints = breakpoints
        self.n_breakpoints = len(self.breakpoints)

        assert isinstance(self.penalizer, float), "penalizer must be a float"
        self.penalizer = penalizer
        self._fitted_parameter_names = ["lambda_%d_" % i for i in range(self.n_breakpoints + 1)]

    def _add_penalty(self, params, neg_ll):
        params_stacked = np.stack(params.values())
        coef_penalty = 0
        if self.penalizer > 0:
            for i in range(params_stacked.shape[1]):
                if not self._cols_to_not_penalize.iloc[i]:
                    coef_penalty = coef_penalty + (params_stacked[:, i]).var()

        return neg_ll + self.penalizer * coef_penalty

    def _cumulative_hazard(self, params, T, Xs):
        n = T.shape[0]
        T = T.reshape((n, 1))
        bps = np.append(self.breakpoints, [np.inf])
        M = np.minimum(np.tile(bps, (n, 1)), T)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])
        lambdas_ = np.array([safe_exp(-np.dot(Xs[param], params[param])) for param in self._fitted_parameter_names])
        return (M * lambdas_.T).sum(1)

    def _prep_inputs_for_prediction_and_return_parameters(self, X):
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X = self.regressors.transform_df(X)["lambda_0_"]

        return np.array([np.exp(np.dot(X, self.params_["lambda_%d_" % i])) for i in range(self.n_breakpoints + 1)])

    def predict_cumulative_hazard(self, df, times=None, conditional_after=None) -> pd.DataFrame:
        """
        Return the cumulative hazard rate of subjects in X at time points.

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved). Uses a linear interpolation if
            points in time are not in the index.

        Returns
        -------
        cumulative_hazard_ : DataFrame
            the cumulative hazard of individuals over the timeline
        """

        if isinstance(df, pd.Series):
            return self.predict_cumulative_hazard(df.to_frame().T)

        if conditional_after is not None:
            raise NotImplementedError()

        times = np.atleast_1d(coalesce(times, self.timeline)).astype(float)
        n = times.shape[0]
        times = times.reshape((n, 1))

        lambdas_ = self._prep_inputs_for_prediction_and_return_parameters(df)

        bp = np.append(self.breakpoints, [np.inf])
        M = np.minimum(np.tile(bp, (n, 1)), times)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])

        return pd.DataFrame(np.dot(M, (1 / lambdas_)), columns=_get_index(df), index=times[:, 0])
