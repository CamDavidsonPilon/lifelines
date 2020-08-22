# -*- coding: utf-8 -*-
from autograd import numpy as np
from autograd.builtins import DictBox
from autograd.numpy.numpy_boxes import ArrayBox
from typing import Dict, List, Optional, Union
from scipy.special import gamma
import pandas as pd

from lifelines.utils import _get_index
from lifelines.fitters import ParametericAFTRegressionFitter
from lifelines.fitters.mixins import ProportionalHazardMixin
from lifelines.utils.safe_exp import safe_exp
from lifelines.utils import DataframeSlicer
from lifelines.statistics import proportional_hazard_test


class WeibullAFTFitter(ParametericAFTRegressionFitter, ProportionalHazardMixin):
    r"""
    This class implements a Weibull AFT model. The model has parameterized
    form, with :math:`\lambda(x) = \exp\left(\beta_0 + \beta_1x_1 + ... + \beta_n x_n \right)`,
    and optionally, :math:`\rho(y) = \exp\left(\alpha_0 + \alpha_1 y_1 + ... + \alpha_m y_m \right)`,

    .. math::  S(t; x, y) = \exp\left(-\left(\frac{t}{\lambda(x)}\right)^{\rho(y)}\right),

    With no covariates, the Weibull model's parameters has the following interpretations: The :math:`\lambda` (scale) parameter has an
    applicable interpretation: it represent the time when 37% of the population has died.
    The :math:`\rho` (shape) parameter controls if the cumulative hazard (see below) is convex or concave, representing accelerating or decelerating
    hazards.

    The cumulative hazard rate is

    .. math:: H(t; x, y) = \left(\frac{t}{\lambda(x)} \right)^{\rho(y)},

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
    _scipy_fit_options = {"ftol": 1e-10, "maxiter": 200}
    _ancillary_parameter_name = "rho_"
    _primary_parameter_name = "lambda_"

    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        model_ancillary: bool = False,
    ) -> None:
        super(WeibullAFTFitter, self).__init__(alpha, penalizer, l1_ratio, fit_intercept, model_ancillary)

    def _cumulative_hazard(
        self, params: Union[DictBox, Dict[str, np.array]], T: Union[float, np.array], Xs: DataframeSlicer
    ) -> Union[np.array, ArrayBox]:
        lambda_params = params["lambda_"]
        log_lambda_ = Xs["lambda_"] @ lambda_params

        rho_params = params["rho_"]
        rho_ = safe_exp(Xs["rho_"] @ rho_params)

        return safe_exp(rho_ * (np.log(np.clip(T, 1e-100, np.inf)) - log_lambda_))

    def _survival_function(
        self, params: Union[DictBox, Dict[str, np.array]], T: Union[float, np.array], Xs: DataframeSlicer
    ) -> Union[np.array, ArrayBox]:
        ch = self._cumulative_hazard(params, T, Xs)
        return safe_exp(-ch)

    def _log_hazard(
        self, params: Union[DictBox, Dict[str, np.array]], T: Union[float, np.array], Xs: DataframeSlicer
    ) -> Union[np.array, ArrayBox]:
        lambda_params = params["lambda_"]
        log_lambda_ = Xs["lambda_"] @ lambda_params

        rho_params = params["rho_"]
        log_rho_ = Xs["rho_"] @ rho_params

        return log_rho_ - log_lambda_ + np.expm1(log_rho_) * (np.log(T) - log_lambda_)

    def predict_percentile(
        self,
        df: pd.DataFrame,
        *,
        ancillary: Optional[pd.DataFrame] = None,
        p: float = 0.5,
        conditional_after: Optional[np.array] = None
    ) -> pd.Series:
        """
        Returns the median lifetimes for the individuals, by default. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions

        Parameters
        ----------
        df:  DataFrame
            a (n,d)  DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary: DataFrame, optional
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
        predict_median, predict_expectation

        """

        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary)

        if conditional_after is None and len(df.shape) == 2:
            conditional_after = np.zeros(df.shape[0])
        elif conditional_after is None and len(df.shape) == 1:
            conditional_after = np.zeros(1)

        return pd.Series(
            lambda_ * np.power(-np.log(p) + (conditional_after / lambda_) ** rho_, 1 / rho_) - conditional_after,
            index=_get_index(df),
        )

    def predict_expectation(self, df: pd.DataFrame, ancillary: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Predict the expectation of lifetimes, :math:`E[T | x]`.

        Parameters
        ----------
        df: DataFrame
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary:  DataFrame, optional
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame
            the expected lifetimes for the individuals.


        See Also
        --------
        predict_median, predict_percentile
        """
        lambda_, rho_ = self._prep_inputs_for_prediction_and_return_scores(df, ancillary)
        return pd.Series((lambda_ * gamma(1 + 1 / rho_)), index=_get_index(df))
