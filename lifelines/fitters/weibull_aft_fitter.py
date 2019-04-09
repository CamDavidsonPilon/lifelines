# -*- coding: utf-8 -*-
from autograd import numpy as np
from scipy.special import gamma
import pandas as pd

from lifelines.utils import _get_index, coalesce
from lifelines.fitters import ParametericAFTRegressionFitter


class WeibullAFTFitter(ParametericAFTRegressionFitter):
    r"""
    This class implements a Weibull AFT model. The model has parameterized
    form, with :math:`\lambda(x) = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and optionally, :math:`\rho(y) = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    .. math::  S(t; x, y) = \exp\left(-\left(\frac{t}{\lambda(x)}\right)^{\rho(y)}\right),

    which implies the cumulative hazard rate is

    .. math:: H(t; x, y) = \left(\frac{t}{\lambda(x)} \right)^{\rho(y)},

    After calling the ``.fit`` method, you have access to properties like:
    ``params_``, ``print_summary()``. A summary of the fit is available with the method ``print_summary()``.


    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    fit_intercept: boolean, optional (default=True)
        Allow lifelines to add an intercept column of 1s to df, and ancillary_df if applicable.

    penalizer: float, optional (default=0.0)
        the penalizer coefficient to the size of the coefficients. See `l1_ratio`. Must be equal to or greater than 0.

    l1_ratio: float, optional (default=0.0)
        how much of the penalizer should be attributed to an l1 penalty (otherwise an l2 penalty). The penalty function looks like
        ``penalizer * l1_ratio * ||w||_1 + 0.5 * penalizer * (1 - l1_ratio) * ||w||^2_2``

    Attributes
    ----------
    params_ : DataFrame
        The estimated coefficients
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the coefficients
    durations: Series
        The event_observed variable provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    standard_errors_: Series
        the standard errors of the estimates
    score_: float
        the concordance index of the model.
    """

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio=0.0, fit_intercept=True):
        self._ancillary_parameter_name = "rho_"
        self._primary_parameter_name = "lambda_"
        super(WeibullAFTFitter, self).__init__(alpha, penalizer, l1_ratio, fit_intercept)

    def _cumulative_hazard(self, params, T, *Xs):
        lambda_params = params[self._LOOKUP_SLICE["lambda_"]]
        lambda_ = np.exp(np.dot(Xs[0], lambda_params))

        rho_params = params[self._LOOKUP_SLICE["rho_"]]
        rho_ = np.exp(np.dot(Xs[1], rho_params))
        return (T / lambda_) ** rho_

    def _log_hazard(self, params, T, *Xs):
        lambda_params = params[self._LOOKUP_SLICE["lambda_"]]
        log_lambda_ = np.dot(Xs[0], lambda_params)

        rho_params = params[self._LOOKUP_SLICE["rho_"]]
        log_rho_ = np.dot(Xs[1], rho_params)

        return log_rho_ - log_lambda_ + np.expm1(log_rho_) * (np.log(T) - log_lambda_)

    def predict_percentile(self, X, ancillary_X=None, p=0.5):
        """
        Returns the median lifetimes for the individuals, by default. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions

        Parameters
        ----------
        X:  numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_X: numpy array or DataFrame, optional
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        p: float, optional (default=0.5)
            the percentile, must be between 0 and 1.

        Returns
        -------
        percentiles: DataFrame

        See Also
        --------
        predict_median

        """
        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(X, ancillary_X)

        return pd.DataFrame(lambda_ * np.power(-np.log(p), 1 / rho_), index=_get_index(X))

    def predict_expectation(self, X, ancillary_X=None):
        """
        Predict the expectation of lifetimes, :math:`E[T | x]`.

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_X: numpy array or DataFrame, optional
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        percentiles: DataFrame
            the median lifetimes for the individuals. If the survival curve of an
            individual does not cross 0.5, then the result is infinity.


        See Also
        --------
        predict_median
        """
        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(X, ancillary_X)
        return pd.DataFrame((lambda_ * gamma(1 + 1 / rho_)), index=_get_index(X))

    def predict_cumulative_hazard(self, X, times=None, ancillary_X=None):
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
        ancillary_X: numpy array or DataFrame, optional
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        cumulative_hazard_ : DataFrame
            the cumulative hazard of individuals over the timeline
        """
        times = coalesce(times, self.timeline, np.unique(self.durations))
        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(X, ancillary_X)

        return pd.DataFrame(np.outer(times, 1 / lambda_) ** rho_, columns=_get_index(X), index=times)
