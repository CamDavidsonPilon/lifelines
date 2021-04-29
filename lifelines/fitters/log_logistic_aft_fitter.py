# -*- coding: utf-8 -*-


from autograd import numpy as np
import pandas as pd

from lifelines.utils import _get_index, coalesce
from lifelines.fitters import ParametericAFTRegressionFitter
from lifelines.utils.safe_exp import safe_exp


class LogLogisticAFTFitter(ParametericAFTRegressionFitter):
    r"""
    This class implements a Log-Logistic AFT model. The model has parameterized
    form, with :math:`\alpha(x) = \exp\left(a_0 + a_1x_1 + ... + a_n x_n \right)`,
    and optionally, :math:`\beta(y) = \exp\left(b_0 + b_1 y_1 + ... + b_m y_m \right)`,


    The cumulative hazard rate is

    .. math:: H(t; x , y) = \log\left(1 + \left(\frac{t}{\alpha(x)}\right)^{\beta(y)}\right)

    The :math:`\alpha` (scale) parameter has an interpretation as being equal to the *median* lifetime. The
    :math:`\beta` parameter influences the shape of the hazard.

    After calling the ``.fit`` method, you have access to properties like:
    ``params_``, ``print_summary()``. A summary of the fit is available with the method ``print_summary()``.


    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    fit_intercept: boolean, optional (default=True)
        Allow lifelines to add an intercept column of 1s to df, and ancillary if applicable.

    penalizer: float or array, optional (default=0.0)
        the penalizer coefficient to the size of the coefficients. See `l1_ratio`. Must be equal to or greater than 0.
        Alternatively, penalizer is an array equal in size to the number of parameters, with penalty coefficients for specific variables. For
        example, `penalizer=0.01 * np.ones(p)` is the same as `penalizer=0.01`


    l1_ratio: float, optional (default=0.0)
        how much of the penalizer should be attributed to an l1 penalty (otherwise an l2 penalty). The penalty function looks like
        ``penalizer * l1_ratio * ||w||_1 + 0.5 * penalizer * (1 - l1_ratio) * ||w||^2_2``

    model_ancillary: optional (default=False)
        set the model instance to always model the ancillary parameter with the supplied Dataframe.
        This is useful for grid-search optimization.

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
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    standard_errors_: Series
        the standard errors of the estimates
    score_: float
        the concordance index of the model.
    """

    # about 25% faster than BFGS
    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"ftol": 1e-6, "maxiter": 200}
    _ancillary_parameter_name = "beta_"
    _primary_parameter_name = "alpha_"

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio=0.0, fit_intercept=True, model_ancillary=False):
        super(LogLogisticAFTFitter, self).__init__(alpha, penalizer, l1_ratio, fit_intercept)

    def _cumulative_hazard(self, params, T, Xs):
        alpha_params = params["alpha_"]
        alpha_ = safe_exp(np.dot(Xs["alpha_"], alpha_params))

        beta_params = params["beta_"]
        beta_ = np.exp(np.dot(Xs["beta_"], beta_params))
        return np.logaddexp(beta_ * (np.log(np.clip(T, 1e-25, np.inf)) - np.log(alpha_)), 0)

    def _log_hazard(self, params, T, Xs):
        alpha_params = params["alpha_"]
        log_alpha_ = np.dot(Xs["alpha_"], alpha_params)
        alpha_ = safe_exp(log_alpha_)

        beta_params = params["beta_"]
        log_beta_ = np.dot(Xs["beta_"], beta_params)
        beta_ = safe_exp(log_beta_)

        return (
            log_beta_
            - log_alpha_
            + np.expm1(log_beta_) * (np.log(T) - log_alpha_)
            - np.logaddexp(beta_ * (np.log(T) - np.log(alpha_)), 0)
        )

    def _log_1m_sf(self, params, T, Xs):
        alpha_params = params["alpha_"]
        log_alpha_ = np.dot(Xs["alpha_"], alpha_params)
        alpha_ = safe_exp(log_alpha_)

        beta_params = params["beta_"]
        log_beta_ = np.dot(Xs["beta_"], beta_params)
        beta_ = safe_exp(log_beta_)
        return -np.logaddexp(-beta_ * (np.log(T) - np.log(alpha_)), 0)

    def predict_percentile(self, df, ancillary=None, p=0.5, conditional_after=None) -> pd.Series:
        """
        Returns the median lifetimes for the individuals, by default. If the survival curve of an
        individual does not cross ``p``, then the result is infinity.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions

        Parameters
        ----------
        X:  DataFrame
            a (n,d)  DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_X: DataFrame, optional
            a (n,d) DataFrame. If a DataFrame, columns
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
        alpha_, beta_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary)

        if conditional_after is None:
            return pd.Series(alpha_ * (1 / (p) - 1) ** (1 / beta_), index=_get_index(df))
        else:
            conditional_after = np.asarray(conditional_after)
            S = 1 / (1 + (conditional_after / alpha_) ** beta_)
            return pd.Series(alpha_ * (1 / (p * S) - 1) ** (1 / beta_) - conditional_after, index=_get_index(df))

    def predict_expectation(self, df, ancillary=None) -> pd.Series:
        """
        Predict the expectation of lifetimes, :math:`E[T | x]`.

        Parameters
        ----------
        X:  DataFrame
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary_X: DataFrame, optional
            a (n,d) DataFrame. If a DataFrame, columns
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
        alpha_, beta_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary)
        v = (alpha_ * np.pi / beta_) / np.sin(np.pi / beta_)
        v = np.where(beta_ > 1, v, np.nan)
        return pd.Series(v, index=_get_index(df))
