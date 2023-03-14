# -*- coding: utf-8 -*-
from autograd import numpy as np
from autograd.scipy.stats import norm
from scipy.special import erfinv
import pandas as pd

from lifelines.utils import _get_index
from lifelines.fitters import ParametericAFTRegressionFitter
from lifelines.utils.safe_exp import safe_exp

from autograd.builtins import DictBox
from autograd.numpy.numpy_boxes import ArrayBox
from lifelines.utils import DataframeSlicer
from numpy import ndarray
from typing import Dict, List, Optional, Union


class LogNormalAFTFitter(ParametericAFTRegressionFitter):
    r"""
    This class implements a Log-Normal AFT model. The model has parameterized
    form, with :math:`\mu(x) = a_0 + a_1x_1 + ... + a_n x_n`,
    and optionally, :math:`\sigma(y) = \exp\left(b_0 + b_1 y_1 + ... + b_m y_m \right)`,

    The cumulative hazard rate is

    .. math:: H(t; x, y) = -\log\left(1 - \Phi\left(\frac{\log(T) - \mu(x)}{\sigma(y)}\right)\right)

    After calling the ``.fit`` method, you have access to properties like:
    ``params_``, ``print_summary()``. A summary of the fit is available with the method ``print_summary()``.


    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    fit_intercept: bool, optional (default=True)
        Allow lifelines to add an intercept column of 1s to df, and ancillary if applicable.

    penalizer: float or array, optional (default=0.0)
        the penalizer coefficient to the size of the coefficients. See `l1_ratio`. Must be equal to or greater than 0.
        Alternatively, penalizer is an array equal in size to the number of parameters, with penalty coefficients for specific variables. For
        example, `penalizer=0.01 * np.ones(p)` is the same as `penalizer=0.01`

    l1_ratio: float, optional (default=0.0)
        how much of the penalizer should be attributed to an l1 penalty (otherwise an l2 penalty). The penalty function looks like
        ``penalizer * l1_ratio * ||w||_1 + 0.5 * penalizer * (1 - l1_ratio) * ||w||^2_2``

    model_ancillary: optional (default=False)
        set the model instance to always model the ancillary parameter with the supplied DataFrame.
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
    _primary_parameter_name = "mu_"
    _ancillary_parameter_name = "sigma_"

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio=0.0, fit_intercept=True, model_ancillary=False):
        super(LogNormalAFTFitter, self).__init__(alpha, penalizer, l1_ratio, fit_intercept, model_ancillary)

    def _cumulative_hazard(
        self, params: Union[DictBox, Dict[str, ndarray]], T: Union[float, ndarray], Xs: DataframeSlicer
    ) -> Union[ndarray, ArrayBox]:
        mu_params = params["mu_"]
        mu_ = np.dot(Xs["mu_"], mu_params)

        sigma_params = params["sigma_"]
        sigma_ = safe_exp(np.dot(Xs["sigma_"], sigma_params))
        Z = (np.log(T) - mu_) / sigma_
        return -norm.logsf(Z)

    def _log_hazard(self, params: DictBox, T: Union[float, ndarray], Xs: DataframeSlicer) -> ArrayBox:
        mu_params = params["mu_"]
        mu_ = np.dot(Xs["mu_"], mu_params)

        sigma_params = params["sigma_"]

        log_sigma_ = np.dot(Xs["sigma_"], sigma_params)
        sigma_ = safe_exp(log_sigma_)
        Z = (np.log(T) - mu_) / sigma_

        return norm.logpdf(Z) - log_sigma_ - np.log(T) - norm.logsf(Z)

    def _log_1m_sf(self, params, T, Xs):
        mu_params = params["mu_"]
        mu_ = np.dot(Xs["mu_"], mu_params)

        sigma_params = params["sigma_"]

        log_sigma_ = np.dot(Xs["sigma_"], sigma_params)
        sigma_ = safe_exp(log_sigma_)
        Z = (np.log(T) - mu_) / sigma_
        return norm.logcdf(Z)

    def predict_percentile(
        self,
        df: pd.DataFrame,
        *,
        ancillary: Optional[pd.DataFrame] = None,
        p: float = 0.5,
        conditional_after: Optional[ndarray] = None
    ) -> pd.Series:
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
        conditional_after: iterable, optional
            Must be equal in size to df.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.


        Returns
        -------
        percentiles: DataFrame

        See Also
        --------
        predict_median

        """
        exp_mu_, sigma_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary)

        if conditional_after is None:
            return pd.Series(exp_mu_ * np.exp(np.sqrt(2) * sigma_ * erfinv(2 * (1 - p) - 1)), index=_get_index(df))
        else:
            conditional_after = np.asarray(conditional_after)
            Z = (np.log(conditional_after) - np.log(exp_mu_)) / sigma_
            S = norm.sf(Z)

            return pd.Series(
                exp_mu_ * np.exp(np.sqrt(2) * sigma_ * erfinv(2 * (1 - p * S) - 1)) - conditional_after, index=_get_index(df)
            )

    def predict_expectation(self, df: pd.DataFrame, ancillary: Optional[pd.DataFrame] = None) -> pd.Series:
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
        exp_mu_, sigma_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary)
        return pd.Series(exp_mu_ * np.exp(sigma_ ** 2 / 2), index=_get_index(df))
