# -*- coding: utf-8 -*-


from autograd import numpy as np
import pandas as pd

from lifelines.utils import _get_index, coalesce
from lifelines.fitters import ParametericAFTRegressionFitter


class LogLogisticAFTFitter(ParametericAFTRegressionFitter):
    r"""
    This class implements a Log-Logistic AFT model. The model has parameterized
    form, with :math:`\alpha(x) = \exp\left(a_0 + a_1x_1 + ... + a_n x_n \right)`,
    and optionally, :math:`\beta(y) = \exp\left(b_0 + b_1 y_1 + ... + b_m y_m \right)`,

    The cumulative hazard rate is

    .. math:: H(t; x , y) = \log\left(1 + \left(\frac{t}{\alpha(x)}\right)^ \beta(y)\right)

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
        self._ancillary_parameter_name = "beta_"
        self._primary_parameter_name = "alpha_"
        super(LogLogisticAFTFitter, self).__init__(alpha, penalizer, l1_ratio, fit_intercept)

    def _cumulative_hazard(self, params, T, *Xs):
        alpha_params = params[self._LOOKUP_SLICE["alpha_"]]
        alpha_ = np.exp(np.dot(Xs[0], alpha_params))

        beta_params = params[self._LOOKUP_SLICE["beta_"]]
        beta_ = np.exp(np.dot(Xs[1], beta_params))
        return np.log1p((T / alpha_) ** beta_)

    def _log_hazard(self, params, T, *Xs):
        alpha_params = params[self._LOOKUP_SLICE["alpha_"]]
        log_alpha_ = np.dot(Xs[0], alpha_params)
        alpha_ = np.exp(log_alpha_)

        beta_params = params[self._LOOKUP_SLICE["beta_"]]
        log_beta_ = np.dot(Xs[1], beta_params)
        beta_ = np.exp(log_beta_)

        return log_beta_ - log_alpha_ + np.expm1(log_beta_) * (np.log(T) - log_alpha_) - np.log1p((T / alpha_) ** beta_)

    def _log_1m_sf(self, params, T, *Xs):
        alpha_params = params[self._LOOKUP_SLICE["alpha_"]]
        log_alpha_ = np.dot(Xs[0], alpha_params)
        alpha_ = np.exp(log_alpha_)

        beta_params = params[self._LOOKUP_SLICE["beta_"]]
        log_beta_ = np.dot(Xs[1], beta_params)
        beta_ = np.exp(log_beta_)
        return -np.log1p((T / alpha_) ** -beta_)

    def predict_percentile(self, X, ancillary_X=None, p=0.5):
        """
        Returns the median lifetimes for the individuals, by default. If the survival curve of an
        individual does not cross ``p``, then the result is infinity.
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
        alpha_, beta_ = self._prep_inputs_for_prediction_and_return_scores(X, ancillary_X)

        return pd.DataFrame(alpha_ * (1 / p - 1) ** beta_, index=_get_index(X))

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
        alpha_, beta_ = self._prep_inputs_for_prediction_and_return_scores(X, ancillary_X)
        v = (alpha_ * np.pi / beta_) / np.sin(np.pi / beta_)
        v = np.where(beta_ > 1, v, np.nan)
        return pd.DataFrame(v, index=_get_index(X))

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
        alpha_, beta_ = self._prep_inputs_for_prediction_and_return_scores(X, ancillary_X)
        return pd.DataFrame(np.log1p(np.outer(times, 1 / alpha_) ** beta_), columns=_get_index(X), index=times)
