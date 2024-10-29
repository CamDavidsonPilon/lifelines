# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Iterator, List, Optional, Tuple, Union, Any, Iterable
from textwrap import dedent, fill
from datetime import datetime
import warnings
import time

from numpy import dot, einsum, log, exp, zeros, arange, multiply, ndarray
import numpy as np
from scipy.linalg import solve as spsolve, LinAlgError, norm, inv
from scipy.integrate import trapezoid
from scipy import stats
from pandas import DataFrame, Series, Index
import pandas as pd
from autograd import elementwise_grad
from autograd import numpy as anp

from lifelines.utils.concordance import _concordance_summary_statistics, _concordance_ratio, concordance_index
from lifelines.fitters import RegressionFitter, SemiParametricRegressionFitter, ParametricRegressionFitter
from lifelines.fitters.mixins import SplineFitterMixin, ProportionalHazardMixin
from lifelines.statistics import _chisq_test_p_value, StatisticalResult
from lifelines.plotting import set_kwargs_drawstyle
from lifelines.utils.safe_exp import safe_exp
from lifelines.utils.printer import Printer
from lifelines import exceptions
from lifelines import utils

__all__ = ["CoxPHFitter"]


class CoxPHFitter(RegressionFitter, ProportionalHazardMixin):
    r"""
    This class implements fitting Cox's proportional hazard model.

    .. math::  h(t|x) = h_0(t) \exp((x - \overline{x})' \beta)

    The baseline hazard, :math:`h_0(t)` can be modeled in two ways:

    1. (default) non-parametrically, using Breslow's method. In this case, the entire model is the traditional semi-parametric Cox model. Ties are
    handled using Efron's method.

    2. parametrically, using a pre-specified number of cubic splines, or piecewise values.

    This is specified using the ``baseline_estimation_method`` parameter in the initialization (default = ``"breslow"``)

    Parameters
    ----------

      alpha: float, optional (default=0.05)
        the level in the confidence intervals.

      baseline_estimation_method: string, optional
        specify how the fitter should estimate the baseline. ``"breslow"``, ``"spline"``, or ``"piecewise"``

      penalizer: float or array, optional (default=0.0)
        Attach a penalty to the size of the coefficients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the magnitude value of :math:`\beta_i`. See ``l1_ratio`` below.
        The penalty term is :math:`\text{penalizer} \left( \frac{1-\text{l1_ratio}}{2} ||\beta||_2^2 + \text{l1_ratio}||\beta||_1\right)`.

        Alternatively, penalizer is an array equal in size to the number of parameters, with penalty coefficients for specific variables. For
        example, `penalizer=0.01 * np.ones(p)` is the same as `penalizer=0.01`

      l1_ratio: float, optional (default=0.0)
        Specify what ratio to assign to a L1 vs L2 penalty. Same as scikit-learn. See ``penalizer`` above.

      strata: list, optional
        specify a list of columns to use in stratification. This is useful if a
        categorical covariate does not obey the proportional hazard assumption. This
        is used similar to the `strata` expression in R.
        See http://courses.washington.edu/b515/l17.pdf.

      n_baseline_knots: int
        Used when ``baseline_estimation_method="spline"``. Set the number of knots (interior & exterior) in the baseline hazard, which will be placed evenly along the time axis.
        Should be at least 2. Royston et. al, the authors of this model, suggest 4 to start, but any values between 2 and 8 are reasonable.
        If you need to customize the timestamps used to calculate the curve, use the ``knots`` parameter instead.

      knots: list, optional
        When ``baseline_estimation_method="spline"``, this allows customizing the points in the time axis for the baseline hazard curve.
        To use evenly-spaced points in time, the ``n_baseline_knots`` parameter can be employed instead.

      breakpoints: list, optional
        Used when ``baseline_estimation_method="piecewise"``. Set the positions of the baseline hazard breakpoints.

    Examples
    --------
    .. code:: python

        from lifelines.datasets import load_rossi
        from lifelines import CoxPHFitter
        rossi = load_rossi()
        cph = CoxPHFitter()
        cph.fit(rossi, 'week', 'arrest')
        cph.print_summary()

    Attributes
    ----------
    params_ : Series
        The estimated coefficients. Changed in version 0.22.0: use to be ``.hazards_``
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    durations: Series
        The durations provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    log_likelihood_: float
        the log-likelihood at the fitted coefficients
    AIC_: float
        the AIC at the fitted coefficients (if using splines for baseline hazard)
    partial_AIC_: float
        the AIC at the fitted coefficients (if using non-parametric inference for baseline hazard)
    baseline_hazard_: DataFrame
        the baseline hazard evaluated at the observed times. Estimated using Breslow's method.
    baseline_cumulative_hazard_: DataFrame
        the baseline cumulative hazard evaluated at the observed times. Estimated using Breslow's method.
    baseline_survival_: DataFrame
        the baseline survival evaluated at the observed times. Estimated using Breslow's method.
    summary: Dataframe
        a Dataframe of the coefficients, p-values, CIs, etc. found in ``print_summary``
    """

    _KNOWN_MODEL = True

    def __init__(
        self,
        baseline_estimation_method: str = "breslow",
        penalizer: Union[float, np.ndarray] = 0.0,
        strata: Optional[Union[List[str], str]] = None,
        l1_ratio: float = 0.0,
        n_baseline_knots: Optional[int] = None,
        knots: Optional[List] = None,
        breakpoints: Optional[List] = None,
        **kwargs,
    ) -> None:

        super(CoxPHFitter, self).__init__(**kwargs)

        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError("l1_ratio parameter must in [0, 1].")

        self.penalizer = penalizer
        self.strata = utils._to_list_or_singleton(strata)
        self.l1_ratio = l1_ratio
        self.baseline_estimation_method = baseline_estimation_method
        if knots is not None:
            if n_baseline_knots is not None:
                raise ValueError("knots and n_baseline_knots are mutually exclusive.")
            self.knots = knots
            self.n_baseline_knots = len(knots)
        else:
            self.n_baseline_knots = n_baseline_knots
            self.knots = None
        self.breakpoints = breakpoints

    @utils.CensoringType.right_censoring
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str] = None,
        event_col: Optional[str] = None,
        show_progress: bool = False,
        initial_point: Optional[ndarray] = None,
        strata: Optional[Union[str, List[str]]] = None,
        weights_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        robust: bool = False,
        batch_mode: Optional[bool] = None,
        timeline: Optional[Iterator] = None,
        formula: str = None,
        entry_col: str = None,
        fit_options: Optional[dict] = None,
    ) -> CoxPHFitter:
        """
        Fit the Cox proportional hazard model to a right-censored dataset. Alias of `fit_right_censoring`.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        weights_col: string, optional
            an optional column in the DataFrame, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use ``robust=True`` to get more accurate standard errors.

        cluster_col: string, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.

        entry_col: str, optional
            a column denoting when a subject entered the study, i.e. left-truncation.

        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the ``strata`` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078

        formula: str, optional
            an Wilkinson formula, like in R and statsmodels, for the right-hand-side. If left as None, all columns not assigned as durations, weights, etc. are used. Uses the library Formulaic for parsing.

        batch_mode: bool, optional
            enabling batch_mode can be faster for datasets with a large number of ties. If left as None, lifelines will choose the best option.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        fit_options: dict, optional
            pass kwargs for the fitting algorithm. For semi-parametric models, this is the Newton-Raphson method (see method _newton_raphson_for_efron_model for kwargs)

        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: ``print_summary``, ``hazards_``, ``confidence_intervals_``, ``baseline_survival_``, etc.


        Examples
        --------
        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E')
            cph.print_summary()
            cph.predict_median(df)

        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'weights': [1.1, 0.5, 2.0, 1.6, 1.2, 4.3, 1.4, 4.5, 3.0, 3.2, 0.4, 6.2],
                'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E', strata=['month', 'age'], robust=True, weights_col='weights')
            cph.print_summary()

        """
        self.strata = utils._to_list_or_singleton(utils.coalesce(strata, self.strata))
        self._model = self._fit_model(
            df,
            duration_col,
            event_col=event_col,
            show_progress=show_progress,
            initial_point=initial_point,
            strata=self.strata,
            weights_col=weights_col,
            cluster_col=cluster_col,
            robust=robust,
            batch_mode=batch_mode,
            timeline=timeline,
            formula=formula,
            entry_col=entry_col,
            fit_options=fit_options,
        )
        return self

    @utils.CensoringType.interval_censoring
    def fit_interval_censoring(
        self,
        df: pd.DataFrame,
        lower_bound_col: str,
        upper_bound_col: str,
        event_col: Optional[str] = None,
        show_progress: bool = False,
        initial_point: Optional[ndarray] = None,
        strata: Optional[Union[str, List[str]]] = None,
        weights_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        robust: bool = False,
        batch_mode: Optional[bool] = None,
        timeline: Optional[Iterator] = None,
        formula: str = None,
        entry_col: str = None,
        fit_options: Optional[dict] = None,
    ) -> CoxPHFitter:
        """
        Fit the Cox proportional hazard model to an interval censored dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        lower_bound_col: string
            the name of the column in DataFrame that contains the lower bounds of the intervals.

        upper_bound_col: string
            the name of the column in DataFrame that contains the upper bounds of the intervals.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, this is inferred based on the upper and lower interval limits (equal
            implies observed death.)

        weights_col: string, optional
            an optional column in the DataFrame, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use ``robust=True`` to get more accurate standard errors.

        cluster_col: string, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.

        entry_col: str, optional
            a column denoting when a subject entered the study, i.e. left-truncation.

        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the ``strata`` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078

        formula: str, optional
            an Wilkinson formula, like in R and statsmodels, for the right-hand-side. If left as None, all columns not assigned as durations, weights, etc. are used.

        batch_mode: bool, optional
            enabling batch_mode can be faster for datasets with a large number of ties. If left as None, lifelines will choose the best option.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: ``print_summary``, ``hazards_``, ``confidence_intervals_``, ``baseline_survival_``, etc.


        Examples
        --------
        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E')
            cph.print_summary()
            cph.predict_median(df)

        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'weights': [1.1, 0.5, 2.0, 1.6, 1.2, 4.3, 1.4, 4.5, 3.0, 3.2, 0.4, 6.2],
                'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E', strata=['month', 'age'], robust=True, weights_col='weights')
            cph.print_summary()

        """
        self.strata = utils._to_list_or_singleton(utils.coalesce(strata, self.strata))
        self._model = self._fit_model(
            df,
            (lower_bound_col, upper_bound_col),
            event_col=event_col,
            show_progress=show_progress,
            initial_point=initial_point,
            strata=self.strata,
            weights_col=weights_col,
            cluster_col=cluster_col,
            robust=robust,
            batch_mode=batch_mode,
            timeline=timeline,
            formula=formula,
            entry_col=entry_col,
            fit_options=fit_options,
        )
        return self

    @utils.CensoringType.left_censoring
    def fit_left_censoring(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str] = None,
        event_col: Optional[str] = None,
        show_progress: bool = False,
        initial_point: Optional[ndarray] = None,
        strata: Optional[Union[str, List[str]]] = None,
        weights_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        robust: bool = False,
        batch_mode: Optional[bool] = None,
        timeline: Optional[Iterator] = None,
        formula: str = None,
        entry_col: str = None,
        fit_options: Optional[dict] = None,
    ) -> CoxPHFitter:
        """
        Fit the Cox proportional hazard model to a left censored dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        weights_col: string, optional
            an optional column in the DataFrame, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use ``robust=True`` to get more accurate standard errors.

        cluster_col: string, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.

        entry_col: str, optional
            a column denoting when a subject entered the study, i.e. left-truncation.

        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the ``strata`` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078

        formula: str, optional
            an Wilkinson formula, like in R and statsmodels, for the right-hand-side. If left as None, all columns not assigned as durations, weights, etc. are used.

        batch_mode: bool, optional
            enabling batch_mode can be faster for datasets with a large number of ties. If left as None, lifelines will choose the best option.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: ``print_summary``, ``hazards_``, ``confidence_intervals_``, ``baseline_survival_``, etc.


        Examples
        --------
        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E')
            cph.print_summary()
            cph.predict_median(df)

        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'weights': [1.1, 0.5, 2.0, 1.6, 1.2, 4.3, 1.4, 4.5, 3.0, 3.2, 0.4, 6.2],
                'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E', strata=['month', 'age'], robust=True, weights_col='weights')
            cph.print_summary()

        """
        self.strata = utils._to_list_or_singleton(utils.coalesce(strata, self.strata))
        self._model = self._fit_model(
            df,
            duration_col,
            event_col=event_col,
            show_progress=show_progress,
            initial_point=initial_point,
            strata=self.strata,
            weights_col=weights_col,
            cluster_col=cluster_col,
            robust=robust,
            batch_mode=batch_mode,
            timeline=timeline,
            formula=formula,
            entry_col=entry_col,
            fit_options=fit_options,
        )
        return self

    def __getattr__(self, attr):
        if attr == "_model":
            raise AttributeError("Must call `fit` first.")

        if hasattr(self._model, attr):
            return getattr(self._model, attr)

        raise AttributeError("%s has no attribute '%s'" % (self._class_name, attr))

    def __dir__(self):
        # pretty hacky - probably a better way
        return self._model.__dir__() + [
            "print_summary",
            "baseline_estimation_method",
            "compute_followup_hazard_ratios",
            "plot_partial_effects_on_outcome",
        ]

    def _fit_model(self, *args, **kwargs):
        if self.baseline_estimation_method == "breslow":
            return self._fit_model_breslow(*args, **kwargs)
        elif self.baseline_estimation_method == "spline":
            return self._fit_model_spline(*args, **kwargs)
        elif self.baseline_estimation_method == "piecewise":
            return self._fit_model_piecewise(*args, **kwargs)
        else:
            raise ValueError("Invalid model estimation.")

    def _fit_model_breslow(self, *args, **kwargs):
        model = SemiParametricPHFitter(
            penalizer=self.penalizer, l1_ratio=self.l1_ratio, strata=self.strata, alpha=self.alpha, label=self._label
        )
        if utils.CensoringType.is_right_censoring(self):
            model.fit(*args, **kwargs)
            return model
        else:
            raise ValueError(
                "Left or interval censoring is not supported for the semi-parametric Cox model. Try changing the baseline estimation method to something else, ex: `CoxPHFitter(baseline_estimation_method='spline').fit_...."
            )

    def _fit_model_piecewise(self, *args, **kwargs):
        df = args[0].copy()
        # handle if they provided a formula or not
        formula = kwargs.pop("formula")

        # handle cluster_col
        if kwargs["cluster_col"] is not None:
            raise ValueError("cluster_col is not available for this baseline estimation method")
        assert self.breakpoints is not None, "breakpoints must be set in initialization."

        # these are not needed, should be popped off.
        kwargs.pop("cluster_col")
        kwargs.pop("batch_mode")

        # handle strata
        strata = utils._to_list_or_singleton(kwargs.pop("strata"))

        if strata is None:
            regressors = {**{"beta_": formula}, **{"log_lambda%d_" % i: "1" for i in range(2, len(self.breakpoints) + 2)}}
            strata_values = None
        elif isinstance(strata, (list, str)):
            strata_namer = ParametricPiecewiseBaselinePHFitter._strata_labeler
            strata = utils._to_list_or_singleton(utils._to_list(strata))

            df = df.set_index(strata).sort_index()

            # how many unique strata values are there?
            strata_values = df.groupby(strata).size().index.tolist()
            regressors = {"beta_": formula}
            for stratum in strata_values:
                regressors.update({strata_namer(stratum, i): "1" for i in range(2, len(self.breakpoints) + 2)})
        else:
            raise ValueError("Wrong type for strata. String, None, or list of strings")

        model = ParametricPiecewiseBaselinePHFitter(
            strata=strata,
            strata_values=strata_values,
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            breakpoints=self.breakpoints,
            alpha=self.alpha,
            label=self._label,
        )
        if utils.CensoringType.is_right_censoring(self):
            model.fit_right_censoring(df, *args[1:], regressors=regressors, **kwargs)
        elif utils.CensoringType.is_left_censoring(self):
            model.fit_left_censoring(df, *args[1:], regressors=regressors, **kwargs)
        elif utils.CensoringType.is_interval_censoring(self):
            lb, ub = args[1]
            model.fit_interval_censoring(df, lb, ub, *args[2:], regressors=regressors, **kwargs)
        return model

    def _fit_model_spline(self, *args, **kwargs):

        df = args[0].copy()

        # handle if they provided a formula or not
        formula = kwargs.pop("formula")

        # handle cluster_col
        if kwargs["cluster_col"] is not None:
            raise ValueError("cluster_col is not available for this baseline estimation method")
        assert self.n_baseline_knots is not None, "n_baseline_knots must be set in initialization."

        # these are not needed, should be popped off.
        kwargs.pop("cluster_col")
        kwargs.pop("batch_mode")

        # handle strata
        strata = utils._to_list_or_singleton(kwargs.pop("strata"))

        if strata is None:
            regressors = {**{"beta_": formula}, **{"phi%d_" % i: "1" for i in range(1, self.n_baseline_knots + 1)}}
            strata_values = None
        elif isinstance(strata, (list, str)):
            strata_namer = ParametricSplinePHFitter._strata_labeler
            strata = utils._to_list_or_singleton(utils._to_list(strata))

            df = df.set_index(strata).sort_index()

            # how many unique strata values are there?
            strata_values = df.groupby(strata).size().index.tolist()
            regressors = {"beta_": formula}
            for stratum in strata_values:
                regressors.update({strata_namer(stratum, i): "1" for i in range(1, self.n_baseline_knots + 1)})
        else:
            raise ValueError("Wrong type for strata. String, None, or list of strings")

        model = ParametricSplinePHFitter(
            strata=strata,
            strata_values=strata_values,
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            n_baseline_knots=self.n_baseline_knots,
            knots=self.knots,
            alpha=self.alpha,
            label=self._label,
        )
        if utils.CensoringType.is_right_censoring(self):
            model.fit_right_censoring(df, *args[1:], regressors=regressors, **kwargs)
        elif utils.CensoringType.is_left_censoring(self):
            model.fit_left_censoring(df, *args[1:], regressors=regressors, **kwargs)
        elif utils.CensoringType.is_interval_censoring(self):
            lb, ub = args[1]
            model.fit_interval_censoring(df, lb, ub, *args[2:], regressors=regressors, **kwargs)
        return model

    def print_summary(self, decimals=2, style=None, columns=None, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        style: string
            {html, ascii, latex}
        columns:
            only display a subset of ``summary`` columns. Default all.
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        # Print information about data first
        justify = utils.string_rjustify(25)

        headers = []

        if utils.CensoringType.is_interval_censoring(self):
            headers.append(("lower bound col", "'%s'" % self.lower_bound_col))
            headers.append(("upper bound col", "'%s'" % self.upper_bound_col))
        else:
            headers.append(("duration col", "'%s'" % self.duration_col))

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if self.entry_col:
            headers.append(("entry col", "'%s'" % self.entry_col))
        if self.cluster_col:
            headers.append(("cluster col", "'%s'" % self.cluster_col))
        if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
            headers.append(("penalizer", self.penalizer))
            headers.append(("l1 ratio", self.l1_ratio))
        if self.robust or self.cluster_col:
            headers.append(("robust variance", True))
        if self.strata:
            headers.append(("strata", self.strata))
        if self.baseline_estimation_method == "spline":
            headers.append(("number of baseline knots", self.n_baseline_knots))
        if self.baseline_estimation_method == "piecewise":
            headers.append(("location of breaks", self.breakpoints))

        headers.extend(
            [
                ("baseline estimation", self.baseline_estimation_method),
                ("number of observations", "{:g}".format(self.weights.sum())),
                ("number of events observed", "{:g}".format(self.weights[self.event_observed > 0].sum())),
                (
                    "partial log-likelihood" if self.baseline_estimation_method == "breslow" else "log-likelihood",
                    "{:.{prec}f}".format(self.log_likelihood_, prec=decimals),
                ),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        footers = []
        sr = self.log_likelihood_ratio_test()

        if self.baseline_estimation_method == "breslow":
            footers.extend(
                [
                    ("Concordance", "{:.{prec}f}".format(self.concordance_index_, prec=decimals)),
                    ("Partial AIC", "{:.{prec}f}".format(self.AIC_partial_, prec=decimals)),
                ]
            )
        elif self.baseline_estimation_method in ["spline", "piecewise"]:
            footers.append(("AIC", "{:.{prec}f}".format(self.AIC_, prec=decimals)))

        footers.extend(
            [
                (
                    "log-likelihood ratio test",
                    "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                ),
                ("-log2(p) of ll-ratio test", "{:.{prec}f}".format(-utils.quiet_log2(sr.p_value), prec=decimals)),
            ]
        )

        p = Printer(self, headers, footers, justify, kwargs, decimals, columns)
        p.print(style=style)

    def compute_followup_hazard_ratios(self, training_df: DataFrame, followup_times: Iterable) -> DataFrame:
        """
        Recompute the hazard ratio at different follow-up times (lifelines handles accounting for updated censoring and updated durations).
        This is useful because we need to remember that the hazard ratio is actually a weighted-average of period-specific hazard ratios.

        Parameters
        ----------

        training_df: pd.DataFrame
            The same dataframe used to train the model
        followup_times: Iterable
            a list/array of follow-up times to recompute the hazard ratio at.


        """
        results = {}
        for t in sorted(followup_times):
            assert t <= training_df[self.duration_col].max(), "all follow-up times must be less than max observed duration"
            df = training_df.copy()
            # if we "rollback" the df to time t, who is dead and who is censored
            df[self.event_col] = (df[self.duration_col] <= t) & df[self.event_col]
            df[self.duration_col] = np.minimum(df[self.duration_col], t)

            model = self.__class__(
                penalizer=self.penalizer,
                l1_ratio=self.l1_ratio,
                strata=self.strata,
                baseline_estimation_method=self.baseline_estimation_method,
                n_baseline_knots=self.n_baseline_knots,
            ).fit(
                df,
                self.duration_col,
                self.event_col,
                weights_col=self.weights_col,
                cluster_col=self.cluster_col,
                entry_col=self.entry_col,
            )
            results[t] = model.hazard_ratios_
        return DataFrame(results).T

    def plot_partial_effects_on_outcome(self, covariates, values, plot_baseline=True, y="survival_function", **kwargs):
        """
        Produces a plot comparing the baseline curve of the model versus
        what happens when a covariate(s) is varied over values in a group. This is useful to compare
        subjects' survival as we vary covariate(s), all else being held equal.

        The baseline curve is equal to the predicted curve at all average values (median for ordinal, and mode for categorical)
        in the original dataset. This same logic is applied to the stratified datasets if ``strata`` was used in fitting.

        Parameters
        ----------
        covariates: string or list
            a string (or list of strings) of the covariate(s) in the original dataset that we wish to vary.
        values: 1d or 2d iterable
            an iterable of the specific values we wish the covariate(s) to take on.
        plot_baseline: bool
            also display the baseline survival, defined as the survival at the mean of the original dataset.
        y: str
            one of "survival_function", or "cumulative_hazard"
        kwargs:
            pass in additional plotting commands.

        Returns
        -------
        ax: matplotlib axis, or list of axis'
            the matplotlib axis that be edited.


        Examples
        ---------
        .. code:: python

            from lifelines import datasets, CoxPHFitter
            rossi = datasets.load_rossi()

            cph = CoxPHFitter().fit(rossi, 'week', 'arrest')
            cph.plot_partial_effects_on_outcome('prio', values=arange(0, 15, 3), cmap='coolwarm')

        .. image:: /images/plot_covariate_example1.png

        .. code:: python

            # multiple variables at once
            cph.plot_partial_effects_on_outcome(['prio', 'paro'], values=[
             [0,  0],
             [5,  0],
             [10, 0],
             [0,  1],
             [5,  1],
             [10, 1]
            ], cmap='coolwarm')

        .. image:: /images/plot_covariate_example2.png

        .. code:: python

            # if you have categorical variables, you can do the following to see the
            # effect of all the categories on one plot.
            cph.plot_partial_effects_on_outcome('categorical_var', values=["A", "B", "C"])


        """
        from matplotlib import pyplot as plt

        covariates = utils._to_list(covariates)
        n_covariates = len(covariates)
        values = np.asarray(values)
        if len(values.shape) == 1:
            values = values[None, :].T

        if n_covariates != values.shape[1]:
            raise ValueError("The number of covariates must equal to second dimension of the values array.")

        for covariate in covariates:
            if covariate not in self._central_values.columns:
                raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        drawstyle = "steps-post" if isinstance(self._model, SemiParametricRegressionFitter) else None
        set_kwargs_drawstyle(kwargs, drawstyle)

        if self.strata is None:
            axes = kwargs.pop("ax", None) or plt.figure().add_subplot(111)
            x_bar = self._central_values
            X = pd.concat([x_bar] * values.shape[0])

            if np.array_equal(np.eye(n_covariates), values) or np.array_equal(
                np.append(np.eye(n_covariates), np.zeros((n_covariates, 1)), axis=1), values
            ):
                X.index = ["%s=1" % c for c in covariates]
            else:
                X.index = [", ".join("%s=%s" % (c, v) for (c, v) in zip(covariates, row)) for row in values]
            for covariate, value in zip(covariates, values.T):
                X[covariate] = value

            # if a column is typeA in the dataset, but the user gives us typeB, we want to cast it. This is
            # most relevant for categoricals.
            X = X.astype(self._central_values.dtypes)

            getattr(self, "predict_%s" % y)(X).plot(ax=axes, **kwargs)

            if plot_baseline:
                getattr(self, "predict_%s" % y)(x_bar).plot(ax=axes, ls=":", color="k", drawstyle=drawstyle)

        else:
            axes = []
            for stratum in self.baseline_survival_.columns:
                ax = plt.figure().add_subplot(1, 1, 1)

                # we turn this into a DF so stratum values that are ints are not converted to floats, etc.
                x_bar = self._central_values.loc[stratum].rename("stratum %s baseline %s" % (str(stratum), y)).to_frame().T

                for name, value in zip(utils._to_list(self.strata), utils._to_tuple(stratum)):
                    x_bar[name] = value

                X = pd.concat([x_bar] * values.shape[0])

                if np.array_equal(np.eye(len(covariates)), values):
                    X.index = ["%s=1" % c for c in covariates]
                else:
                    X.index = [", ".join("%s=%s" % (c, v) for (c, v) in zip(covariates, row)) for row in values]

                for covariate, value in zip(covariates, values.T):
                    X[covariate] = value

                # if a column is typeA in the dataset, but the user gives us typeB, we want to cast it. This is
                # most relevant for categoricals.
                X = X.astype(self._central_values.dtypes)

                getattr(self, "predict_%s" % y)(X).plot(ax=ax, **kwargs)
                if plot_baseline:
                    getattr(self, "predict_%s" % y)(x_bar).plot(ax=ax, ls=":", drawstyle=drawstyle)
                plt.legend()
                axes.append(ax)
        return axes


class SemiParametricPHFitter(ProportionalHazardMixin, SemiParametricRegressionFitter):
    r"""
    This class implements fitting Cox's proportional hazard model using Efron's method for ties.

    .. math::  h(t|x) = h_0(t) \exp((x - \overline{x})' \beta)

    The baseline hazard, :math:`h_0(t)` is modeled non-parametrically (using Breslow's method).

    Note
    -------
    This is a "hidden" class that is invoked when using ``baseline_estimation_method="breslow"`` (the default). You probably want to use ``CoxPHFitter``, not this.

    Parameters
    ----------

      alpha: float, optional (default=0.05)
        the level in the confidence intervals.

      penalizer: float or array, optional (default=0.0)
        Attach a penalty to the size of the coefficients during regression. This improves
        stability of the estimates and controls for high correlation between covariates.
        For example, this shrinks the magnitude value of :math:`\beta_i`. See ``l1_ratio`` below.
        The penalty term is :math:`\text{penalizer} \left( \frac{1-\text{l1_ratio}}{2} ||\beta||_2^2 + \text{l1_ratio}||\beta||_1\right)`.

        Alternatively, penalizer is an array equal in size to the number of parameters, with penalty coefficients for specific variables. For
        example, `penalizer=0.01 * np.ones(p)` is the same as `penalizer=0.01`

      l1_ratio: float, optional (default=0.0)
        Specify what ratio to assign to a L1 vs L2 penalty. Same as scikit-learn. See ``penalizer`` above.

      strata: list, optional
        specify a list of columns to use in stratification. This is useful if a
        categorical covariate does not obey the proportional hazard assumption. This
        is used similar to the `strata` expression in R.
        See http://courses.washington.edu/b515/l17.pdf.

    Examples
    --------
    .. code:: python

        from lifelines.datasets import load_rossi
        from lifelines import CoxPHFitter
        rossi = load_rossi()
        cph = CoxPHFitter()
        cph.fit(rossi, 'week', 'arrest')
        cph.print_summary()

    Attributes
    ----------
    params_ : Series
        The estimated coefficients. Changed in version 0.22.0: use to be ``.hazards_``
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    durations: Series
        The durations provided
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    baseline_hazard_: DataFrame
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """
    _KNOWN_MODEL = True

    def __init__(
        self,
        penalizer: Union[float, np.ndarray] = 0.0,
        strata: Optional[Union[List[str], str]] = None,
        l1_ratio: float = 0.0,
        **kwargs,
    ) -> None:

        super(SemiParametricPHFitter, self).__init__(**kwargs)

        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError("l1_ratio parameter must in [0, 1].")

        self.penalizer = penalizer
        self.strata = utils._to_list_or_singleton(strata)
        self.l1_ratio = l1_ratio

    @utils.CensoringType.right_censoring
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str] = None,
        event_col: Optional[str] = None,
        show_progress: bool = False,
        initial_point: Optional[ndarray] = None,
        strata: Optional[Union[str, List[str]]] = None,
        weights_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        robust: bool = False,
        batch_mode: Optional[bool] = None,
        timeline: Optional[Iterator] = None,
        formula: str = None,
        entry_col: str = None,
        fit_options: Optional[dict] = None,
    ) -> SemiParametricPHFitter:
        """
        Fit the Cox proportional hazard model to a dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        weights_col: string, optional
            an optional column in the DataFrame, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use ``robust=True`` to get more accurate standard errors.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the ``strata`` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078

        cluster_col: string, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.

        batch_mode: bool, optional
            enabling batch_mode can be faster for datasets with a large number of ties. If left as None, lifelines will choose the best option.

        fit_options: dict, optional
            Override the default values in NR algorithm:
                step_size: 0.95,
                precision: 1e-07,
                max_steps: 500,

        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: ``print_summary``, ``hazards_``, ``confidence_intervals_``, ``baseline_survival_``, etc.


        Note
        ----
        Tied survival times are handled using Efron's tie-method.


        Examples
        --------
        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E')
            cph.print_summary()
            cph.predict_median(df)

        .. code:: python

            from lifelines import CoxPHFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'weights': [1.1, 0.5, 2.0, 1.6, 1.2, 4.3, 1.4, 4.5, 3.0, 3.2, 0.4, 6.2],
                'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            cph = CoxPHFitter()
            cph.fit(df, 'T', 'E', strata=['month', 'age'], robust=True, weights_col='weights')
            cph.print_summary()
            cph.predict_median(df)

        """
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self.robust = robust
        self.cluster_col = cluster_col
        self.weights_col = weights_col
        self._n_examples = df.shape[0]
        self._batch_mode = batch_mode
        self.strata = utils._to_list_or_singleton(utils.coalesce(strata, self.strata))
        self.formula = formula
        self.entry_col = entry_col

        X, T, E, weights, entries, original_index, self._clusters = self._preprocess_dataframe(df)

        self.durations = T.copy()
        self.event_observed = E.copy()
        self.weights = weights.copy()
        self.entries = entries

        if self.strata is not None:
            self.durations.index = original_index
            self.event_observed.index = original_index
            self.weights.index = original_index

        # TODO: doesn't handle weights, nor strata
        self._central_values = self._compute_central_values_of_raw_training_data(df, self.strata)

        self._norm_mean = X.mean(0)
        self._norm_std = X.std(0)

        # this is surprisingly faster to do...
        X_norm = pd.DataFrame(
            utils.normalize(X.values, self._norm_mean.values, self._norm_std.values), index=X.index, columns=X.columns
        )

        params_, ll_, variance_matrix_, baseline_hazard_, baseline_cumulative_hazard_, model = self._fit_model(
            X_norm,
            T,
            E,
            weights=weights,
            entries=entries,
            fit_options=utils.coalesce(fit_options, dict()),
            initial_point=initial_point,
            show_progress=show_progress,
        )

        self.log_likelihood_ = ll_
        self.model = model
        self.variance_matrix_ = variance_matrix_
        self.params_ = pd.Series(params_, index=pd.Index(X.columns, name="covariate"), name="coef")
        self.baseline_hazard_ = baseline_hazard_
        self.baseline_cumulative_hazard_ = baseline_cumulative_hazard_
        self.timeline = utils.coalesce(timeline, self.baseline_cumulative_hazard_.index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._predicted_partial_hazards_ = (
                self.predict_partial_hazard(df)
                .to_frame(name="P")
                .reindex(original_index)
                .assign(T=self.durations, E=self.event_observed, W=self.weights)
                .set_index(X.index)
            )

        self.standard_errors_ = self._compute_standard_errors(X_norm, T, E, weights)
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_survival_ = self._compute_baseline_survival()

        if hasattr(self, "_concordance_index_"):
            del self._concordance_index_

        return self

    def _preprocess_dataframe(
        self, df: DataFrame
    ) -> Tuple[DataFrame, Series, Series, Series, Optional[Series], Index, Optional[Series]]:
        # this should be a pure function

        df = df.copy()

        if self.strata is not None:
            sort_by = utils._to_list(self.strata) + (
                [self.duration_col, self.event_col] if self.event_col else [self.duration_col]
            )
            df = df.sort_values(by=sort_by)
            original_index = df.index.copy()
            df = df.set_index(self.strata)
        else:
            sort_by = [self.duration_col, self.event_col] if self.event_col else [self.duration_col]
            df = df.sort_values(by=sort_by)
            original_index = df.index.copy()

        # Extract time, event and metadata
        T = df.pop(self.duration_col)
        E = (
            df.pop(self.event_col)
            if (self.event_col is not None)
            else pd.Series(np.ones(self._n_examples), index=df.index, name="E")
        )
        W = (
            df.pop(self.weights_col)
            if (self.weights_col is not None)
            else pd.Series(np.ones((self._n_examples,)), index=df.index, name="weights")
        )
        entries = df.pop(self.entry_col) if (self.entry_col is not None) else None

        _clusters = df.pop(self.cluster_col).values if self.cluster_col else None

        self.regressors = utils.CovariateParameterMappings({"beta_": self.formula}, df, force_no_intercept=True)
        X = self.regressors.transform_df(df)["beta_"]

        T = T.astype(float)

        # we check nans here because converting to bools maps NaNs to True..
        utils.check_nans_or_infs(E)
        E = E.astype(bool)

        self._check_values_pre_fitting(X, T, E, W, entries)

        return X, T, E, W, entries, original_index, _clusters

    def _check_values_post_fitting(self, X, T, E, W):
        """
        Functions here check why a fit may have non-obviously failed
        """
        utils.check_dimensions(X)
        utils.check_complete_separation(X, E, T, self.event_col)

    def _check_values_pre_fitting(self, X, T, E, W, entries):
        """
        Some utilities to check for bad data coming in, like NaNs or complete separation.
        """
        utils.check_low_var(X)
        utils.check_for_numeric_dtypes_or_raise(X)
        utils.check_nans_or_infs(T)
        utils.check_nans_or_infs(X)
        # check to make sure their weights are okay
        if self.weights_col:
            if (W.astype(int) != W).any() and not self.robust:
                warnings.warn(
                    """It appears your weights are not integers, possibly propensity or sampling scores then?
It's important to know that the naive variance estimates of the coefficients are biased. Instead a) set `robust=True` in the call to `fit`, or b) use Monte Carlo to
estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"
""",
                    exceptions.StatisticalWarning,
                )
            if (W <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

        if self.entry_col:
            utils.check_entry_times(T, entries)

    def _fit_model(
        self,
        X: DataFrame,
        T: Series,
        E: Series,
        weights: Series,
        entries: Optional[Series],
        fit_options: dict,
        initial_point: Optional[ndarray] = None,
        show_progress: bool = True,
    ):
        beta_, ll_, hessian_ = self._newton_raphson_for_efron_model(
            X,
            T,
            E,
            weights,
            entries,
            initial_point=initial_point,
            show_progress=show_progress,
            **fit_options,
        )

        # compute the baseline hazard here.
        predicted_partial_hazards_ = (
            pd.DataFrame(np.exp(dot(X, beta_)), columns=["P"]).assign(T=T.values, E=E.values, W=weights.values).set_index(X.index)
        )
        baseline_hazard_ = self._compute_baseline_hazards(predicted_partial_hazards_)
        baseline_cumulative_hazard_ = self._compute_baseline_cumulative_hazard(baseline_hazard_)

        # rescale parameters back to original scale.
        params_ = beta_ / self._norm_std.values
        if hessian_.size > 0:
            # possible if the df is trivial (no covariate columns)
            variance_matrix_ = pd.DataFrame(
                -inv(hessian_) / np.outer(self._norm_std, self._norm_std), index=X.columns, columns=X.columns
            )
        else:
            variance_matrix_ = pd.DataFrame(index=X.columns, columns=X.columns)

        return params_, ll_, variance_matrix_, baseline_hazard_, baseline_cumulative_hazard_, None

    def _choose_gradient_calculator(self, T, X, entries):

        if entries is not None:
            from lifelines import CoxTimeVaryingFitter

            return lambda X, T, E, weights, entries, beta: CoxTimeVaryingFitter._get_gradients(
                X.values, E.values, entries.values, T.values, weights.values, beta
            )

        decision = _BatchVsSingle().decide(self._batch_mode, T.nunique(), *X.shape)
        return getattr(self, "_get_efron_values_%s" % decision)

    def _newton_raphson_for_efron_model(
        self,
        X: DataFrame,
        T: Series,
        E: Series,
        weights: Series,
        entries: Optional[Series],
        initial_point: Optional[ndarray] = None,
        show_progress: bool = True,
        step_size: float = 0.95,
        precision: float = 1e-07,
        r_precision: float = 1e-9,
        max_steps: int = 500,
    ):  # pylint: disable=too-many-statements,too-many-branches
        """
        Newton Raphson algorithm for fitting CPH model.

        Note
        ----
        The data is assumed to be sorted on T!

        Parameters
        ----------
        X: (n,d) Pandas DataFrame of observations.
        T: (n) Pandas Series representing observed durations.
        E: (n) Pandas Series representing death events.
        weights: (n) an iterable representing weights per observation.
        initial_point: (d,) numpy array of initial starting point for
                      NR algorithm. Default 0.
        step_size: float, optional
            > 0.001 to determine a starting step size in NR algorithm.
        precision: float, optional
            the algorithm stops if the norm of delta between
            successive positions is less than ``precision``.
        r_precision: float, optional
            the algorithms stops if the relative decrease in log-likelihood
            between successive iterations goes below ``r_precision``.
        show_progress: bool, optional
            since the fitter is iterative, show convergence diagnostics.
        max_steps: int, optional
            the maximum number of iterations of the Newton-Raphson algorithm.

        Returns
        -------
        beta: (1,d) numpy array.
        """
        CONVERGENCE_DOCS = "Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model"

        n, d = X.shape

        # soft penalizer functions, from https://www.cs.ubc.ca/cgi-bin/tr/2009/TR-2009-19.pdf
        soft_abs = lambda x, a: 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))
        elastic_net_penalty = (
            lambda beta, a: n
            * (self.penalizer * (self.l1_ratio * (soft_abs(beta, a)) + 0.5 * (1 - self.l1_ratio) * (beta**2))).sum()
        )
        d_elastic_net_penalty = elementwise_grad(elastic_net_penalty)
        dd_elastic_net_penalty = elementwise_grad(d_elastic_net_penalty)

        get_gradients = self._choose_gradient_calculator(T, X, entries)

        # make sure betas are correct size.
        if initial_point is not None:
            assert initial_point.shape == (d,)
            beta = initial_point
        else:
            beta = zeros((d,))

        step_sizer = utils.StepSizer(step_size)
        step_size = step_sizer.next()

        delta = np.zeros_like(beta)
        converging = True
        ll_, previous_ll_ = 0.0, 0.0
        start = time.time()
        i = 0

        while converging:
            beta += step_size * delta

            i += 1

            if self.strata is None:

                h, g, ll_ = get_gradients(X, T, E, weights, entries, beta)

            else:
                g = np.zeros_like(beta)
                h = zeros((beta.shape[0], beta.shape[0]))
                ll_ = 0
                for _h, _g, _ll in self._partition_by_strata_and_apply(X, T, E, weights, entries, get_gradients, beta):
                    g += _g
                    h += _h
                    ll_ += _ll

            if i == 1 and np.all(beta == 0):
                # this is a neat optimization, the null partial likelihood
                # is the same as the full partial but evaluated at zero.
                # if the user supplied a non-trivial initial point, we need to delay this.
                self._ll_null_ = ll_

            if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
                ll_ -= elastic_net_penalty(beta, 1.3**i)
                g -= d_elastic_net_penalty(beta, 1.3**i)
                h[np.diag_indices(d)] -= dd_elastic_net_penalty(beta, 1.3**i)

            # reusing a piece to make g * inv(h) * g.T faster later
            try:
                inv_h_dot_g_T = spsolve(-h, g, assume_a="pos", check_finite=False)
            except (ValueError, LinAlgError) as e:
                self._check_values_post_fitting(X, T, E, weights)
                if "infs or NaNs" in str(e):
                    raise exceptions.ConvergenceError(
                        """Hessian or gradient contains nan or inf value(s). Convergence halted. {0}""".format(CONVERGENCE_DOCS),
                        e,
                    )
                elif isinstance(e, LinAlgError):
                    raise exceptions.ConvergenceError(
                        """Convergence halted due to matrix inversion problems. Suspicion is high collinearity. {0}""".format(
                            CONVERGENCE_DOCS
                        ),
                        e,
                    )
                else:
                    # something else?
                    raise e

            delta = inv_h_dot_g_T

            if np.any(np.isnan(delta)):
                self._check_values_post_fitting(X, T, E, weights)
                raise exceptions.ConvergenceError(
                    """delta contains nan value(s). Convergence halted. {0}""".format(CONVERGENCE_DOCS)
                )

            # Save these as pending result
            hessian, gradient = h, g

            if delta.size > 0:
                norm_delta = norm(delta)
            else:
                norm_delta = 0

            # reusing an above piece to make g * inv(h) * g.T faster.
            newton_decrement = g.dot(inv_h_dot_g_T) / 2

            if show_progress:
                print(
                    "\rIteration %d: norm_delta = %.2e, step_size = %.4f, log_lik = %.5f, newton_decrement = %.2e, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll_, newton_decrement, time.time() - start)
                )

            # convergence criteria
            if norm_delta < precision:
                converging, success = False, True
            elif previous_ll_ != 0 and abs(ll_ - previous_ll_) / (-previous_ll_) < r_precision:
                # this is what R uses by default, with r_precision = 1e-9
                converging, success = False, True
            elif newton_decrement < precision:
                converging, success = False, True
            elif i >= max_steps:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is ~10 steps
                converging, success = False, False
            elif step_size <= 0.00001:
                converging, success = False, False
            elif abs(ll_) < 0.0001 and norm_delta > 1.0:
                warnings.warn(
                    "The log-likelihood is getting suspiciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
See https://stats.stackexchange.com/q/11109/11867 for more.\n",
                    exceptions.ConvergenceWarning,
                )
                converging, success = False, False

            previous_ll_ = ll_
            step_size = step_sizer.update(norm_delta).next()

        if show_progress and success:
            print("Convergence success after %d iterations." % (i))
        elif show_progress and not success:
            print("Convergence failed. See any warning messages.")

        # report to the user problems that we detect.
        if success and norm_delta > 0.1:
            self._check_values_post_fitting(X, T, E, weights)
            warnings.warn(
                "Newton-Raphson convergence completed successfully but norm(delta) is still high, %.3f. This may imply non-unique solutions to the maximum likelihood. Perhaps there is collinearity or complete separation in the dataset?\n"
                % norm_delta,
                exceptions.ConvergenceWarning,
            )
        elif not success:
            self._check_values_post_fitting(X, T, E, weights)
            warnings.warn(
                "Newton-Raphson failed to converge sufficiently. {0}".format(CONVERGENCE_DOCS), exceptions.ConvergenceWarning
            )

        return beta, ll_, hessian

    def _get_efron_values_single(
        self, X: DataFrame, T: Series, E: Series, weights: Series, entries: None, beta: ndarray
    ) -> Tuple[ndarray, ndarray, float]:
        """
        Calculates the first and second order vector differentials, with respect to beta.
        Note that X, T, E are assumed to be sorted on T!

        A good explanation for Efron. Consider three of five subjects who fail at the same time.
        As it is not known a priori that who is the first to fail, so one-third of
        (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
        of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

        From https://cran.r-project.org/web/packages/survival/survival.pdf:

        "Setting all weights to 2 for instance will give the same coefficient estimate but halve the variance. When
        the Efron approximation for ties (default) is employed replication of the data will not give exactly the same coefficients as the
        weights option, and in this case the weighted fit is arguably the correct one."

        Parameters
        ----------
        X: array
            (n,d) numpy array of observations.
        T: array
            (n) numpy array representing observed durations.
        E: array
            (n) numpy array representing death events.
        weights: array
            (n) an array representing weights per observation.
        beta: array
            (1, d) numpy array of coefficients.

        Returns
        -------
        hessian:
            (d, d) numpy array,
        gradient:
            (1, d) numpy array
        log_likelihood: float
        """

        X = X.values
        T = T.values
        E = E.values
        weights = weights.values

        n, d = X.shape
        hessian = zeros((d, d))
        gradient = zeros((d,))
        log_lik = 0

        # Init risk and tie sums to zero
        x_death_sum = zeros((d,))
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = zeros((d,)), zeros((d,))
        risk_phi_x_x, tie_phi_x_x = zeros((d, d)), zeros((d, d))

        # Init number of ties and weights
        weight_count = 0.0
        tied_death_counts = 0
        scores = weights * exp(dot(X, beta))

        phi_x_is = scores[:, None] * X
        phi_x_x_i = np.empty((d, d))

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]
            xi = X[i]
            w = weights[i]

            # Calculate phi values
            phi_i = scores[i]
            phi_x_i = phi_x_is[i]
            # https://stackoverflow.com/a/51481295/1895939
            phi_x_x_i = multiply.outer(xi, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = risk_phi + phi_i
            risk_phi_x = risk_phi_x + phi_x_i
            risk_phi_x_x = risk_phi_x_x + phi_x_x_i

            # Calculate sums of Ties, if this is an event
            if ei:
                x_death_sum = x_death_sum + w * xi
                tie_phi = tie_phi + phi_i
                tie_phi_x = tie_phi_x + phi_x_i
                tie_phi_x_x = tie_phi_x_x + phi_x_x_i

                # Keep track of count
                tied_death_counts += 1
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tied_death_counts == 0:
                # Only censored with current time, move on
                continue

            # There was at least one event and no more ties remain. Time to sum.
            # This code is near identical to the _batch algorithm below. In fact, see _batch for comments.
            weighted_average = weight_count / tied_death_counts

            if tied_death_counts > 1:
                increasing_proportion = arange(tied_death_counts) / tied_death_counts
                denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)
                numer = risk_phi_x - multiply.outer(increasing_proportion, tie_phi_x)
                a1 = einsum("ab,i->ab", risk_phi_x_x, denom) - einsum("ab,i->ab", tie_phi_x_x, increasing_proportion * denom)
            else:
                denom = 1.0 / np.array([risk_phi])
                numer = risk_phi_x
                a1 = risk_phi_x_x * denom

            summand = numer * denom[:, None]
            a2 = summand.T.dot(summand)

            gradient = gradient + x_death_sum - weighted_average * summand.sum(0)

            log_lik = log_lik + dot(x_death_sum, beta) + weighted_average * log(denom).sum()
            hessian = hessian + weighted_average * (a2 - a1)

            # reset tie values
            tied_death_counts = 0
            weight_count = 0.0
            x_death_sum = zeros((d,))
            tie_phi = 0
            tie_phi_x = zeros((d,))
            tie_phi_x_x = zeros((d, d))

        return hessian, gradient, log_lik

    def _get_efron_values_batch(
        self, X: DataFrame, T: Series, E: Series, weights: Series, entries: None, beta: ndarray
    ) -> Tuple[ndarray, ndarray, float]:  # pylint: disable=too-many-locals
        """
        Assumes sorted on ascending on T
        Calculates the first and second order vector differentials, with respect to beta.

        A good explanation for how Efron handles ties. Consider three of five subjects who fail at the time.
        As it is not known a priori that who is the first to fail, so one-third of
        (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
        of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

        Returns
        -------
        hessian: (d, d) numpy array,
        gradient: (1, d) numpy array
        log_likelihood: float
        """
        X = X.values
        T = T.values
        E = E.values
        weights = weights.values

        n, d = X.shape
        hessian = zeros((d, d))
        gradient = zeros((d,))
        log_lik = 0
        # weights = weights[:, None]

        # Init risk and tie sums to zero
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = zeros((d,)), zeros((d,))
        risk_phi_x_x, tie_phi_x_x = zeros((d, d)), zeros((d, d))

        # counts are sorted by -T
        _, counts = np.unique(-T, return_counts=True)
        scores = weights * exp(dot(X, beta))
        pos = n
        ZERO_TO_N = arange(counts.max())

        for count_of_removals in counts:

            slice_ = slice(pos - count_of_removals, pos)

            X_at_t = X[slice_]
            weights_at_t = weights[slice_]
            tied_death_counts = E[slice_].sum()

            phi_i = scores[slice_, None]
            phi_x_i = phi_i * X_at_t
            phi_x_x_i = dot(X_at_t.T, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = risk_phi + phi_i.sum()
            risk_phi_x = risk_phi_x + (phi_x_i).sum(0)
            risk_phi_x_x = risk_phi_x_x + phi_x_x_i

            # Calculate the sums of Tie set
            if tied_death_counts == 0:
                # no deaths, can continue
                pos -= count_of_removals
                continue

            # corresponding deaths are sorted like [False, False, False, ..., True, ....] because we sorted early in preprocessing
            xi_deaths = X_at_t[-tied_death_counts:]
            weights_deaths = weights_at_t[-tied_death_counts:]

            x_death_sum = einsum("a,ab->b", weights_deaths, xi_deaths)

            weight_count = weights_deaths.sum()
            weighted_average = weight_count / tied_death_counts

            if tied_death_counts > 1:

                # a lot of this is now in Einstein notation for performance, but see original "expanded" code here
                # https://github.com/CamDavidsonPilon/lifelines/blob/e7056e7817272eb5dff5983556954f56c33301b1/lifelines/fitters/coxph_fitter.py#L755-L789

                # it's faster if we can skip computing these when we don't need to.
                phi_x_i_deaths = phi_x_i[-tied_death_counts:]
                tie_phi = phi_i[-tied_death_counts:].sum()
                tie_phi_x = (phi_x_i_deaths).sum(0)
                tie_phi_x_x = dot(xi_deaths.T, phi_x_i_deaths)

                increasing_proportion = ZERO_TO_N[:tied_death_counts] / tied_death_counts
                numer = risk_phi_x - multiply.outer(increasing_proportion, tie_phi_x)
                denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)

                # computes outer products and sums them together.
                # Naive approach is to
                # 1) broadcast tie_phi_x_x and increasing_proportion into a (tied_death_counts, d, d) matrix
                # 2) broadcast risk_phi_x_x and denom into a (tied_death_counts, d, d) matrix
                # 3) subtract them, and then sum to (d, d)
                # Alternatively, we can sum earlier without having to explicitly create (_, d, d) matrices. This is used here.
                #
                a1 = einsum("ab,i->ab", risk_phi_x_x, denom) - einsum("ab,i->ab", tie_phi_x_x, increasing_proportion * denom)
            else:
                # no tensors here, but do some casting to make it easier in the converging step next.
                numer = risk_phi_x
                denom = 1.0 / np.array([risk_phi])
                a1 = risk_phi_x_x * denom

            summand = numer * denom[:, None]
            # This is a batch outer product.
            # given a matrix t, for each row, m, compute it's outer product: m.dot(m.T), and stack these new matrices together.
            # which would be: einsum("Bi, Bj->Bij", t, t)
            a2 = dot(summand.T, summand)

            gradient = gradient + x_death_sum - weighted_average * summand.sum(0)
            log_lik = log_lik + dot(x_death_sum, beta) + weighted_average * log(denom).sum()
            hessian = hessian + weighted_average * (a2 - a1)
            pos -= count_of_removals

        return hessian, gradient, log_lik

    def _partition_by_strata(self, X: DataFrame, T: Series, E: Series, entries: Optional[Series], weights: Series):
        for stratum, stratified_X in X.groupby(self.strata):
            if entries is None:
                stratified_entries = None
            else:
                stratified_entries = entries.loc[[stratum]]
            stratified_E, stratified_T, stratified_W = (E.loc[[stratum]], T.loc[[stratum]], weights.loc[[stratum]])
            yield (stratified_X, stratified_T, stratified_E, stratified_W, stratified_entries), stratum

    def _partition_by_strata_and_apply(
        self, X: DataFrame, T: Series, E: Series, entries: Optional[Series], weights: Series, function: Callable, *args
    ):
        for (stratified_X, stratified_T, stratified_E, stratified_W, stratified_entries), _ in self._partition_by_strata(
            X, T, E, weights, entries
        ):
            yield function(stratified_X, stratified_T, stratified_E, stratified_W, stratified_entries, *args)

    def _compute_martingale(
        self, X: DataFrame, T: Series, E: Series, _weights: Series, index: Optional[Index] = None
    ) -> pd.DataFrame:
        # TODO: _weights unused
        partial_hazard = self.predict_partial_hazard(X).values

        if not self.strata:
            baseline_at_T = self.baseline_cumulative_hazard_.loc[T, "baseline cumulative hazard"].values
        else:
            baseline_at_T = np.empty(0)
            for name, T_ in T.groupby(by=self.strata):
                baseline_at_T = np.append(baseline_at_T, self.baseline_cumulative_hazard_[name].loc[T_])

        martingale = E - (partial_hazard * baseline_at_T)
        return pd.DataFrame({self.duration_col: T.values, self.event_col: E.values, "martingale": martingale.values}, index=index)

    def _compute_deviance(
        self, X: DataFrame, T: Series, E: Series, weights: Series, index: Optional[Index] = None
    ) -> pd.DataFrame:
        df = self._compute_martingale(X, T, E, weights, index)
        rmart = df.pop("martingale")

        log_term = np.where((E.values - rmart.values) <= 0, 0, E.values * log(E.values - rmart.values))

        deviance = np.sign(rmart) * np.sqrt(-2 * (rmart + log_term))
        df["deviance"] = deviance
        return df

    def _compute_scaled_schoenfeld(
        self, X: DataFrame, T: Series, E: Series, weights: Series, index: Optional[Index] = None
    ) -> pd.DataFrame:
        r"""
        Let s_k be the kth schoenfeld residuals. Then E[s_k] = 0.
        For tests of proportionality, we want to test if \beta_i(t) is \beta_i (constant) or not.

        Let V_k be the contribution to the information matrix at time t_k. A main result from Grambsch and Therneau is that

        \beta(t) = E[s_k*V_k^{-1} + \hat{beta}]

        so define s_k^* = s_k*V_k^{-1} + \hat{beta} as the scaled schoenfeld residuals.

        We can approximate V_k with Hessian/d, so the inverse of Hessian/d is (d * variance_matrix_)

        Notes
        -------
        lifelines does not add the coefficients to the final results, but R does when you call residuals(c, "scaledsch")


        """

        n_deaths = self.event_observed.sum()
        scaled_schoenfeld_resids = n_deaths * self._compute_schoenfeld(X, T, E, weights, index).dot(self.variance_matrix_)

        scaled_schoenfeld_resids.columns = self.params_.index
        return scaled_schoenfeld_resids

    def _compute_schoenfeld(
        self, X: pd.DataFrame, T: pd.Series, E: pd.Series, weights: pd.Series, index: pd.Index
    ) -> pd.DataFrame:
        # TODO: should the index by times, i.e. T[E]?

        # Assumes sorted on T and on strata
        # cluster does nothing to this, as expected.

        _, d = X.shape

        if self.strata is not None:
            schoenfeld_residuals = np.empty((0, d))

            for schoenfeld_residuals_in_strata in self._partition_by_strata_and_apply(
                X, T, E, weights, None, self._compute_schoenfeld_within_strata
            ):
                schoenfeld_residuals = np.append(schoenfeld_residuals, schoenfeld_residuals_in_strata, axis=0)

        else:
            schoenfeld_residuals = self._compute_schoenfeld_within_strata(X, T, E, weights, None)

        # schoenfeld residuals are only defined for subjects with a non-zero event.
        df = pd.DataFrame(schoenfeld_residuals[E, :], columns=self.params_.index, index=index[E])
        return df

    def _compute_schoenfeld_within_strata(self, X: DataFrame, T: Series, E: Series, weights: Series, entries: None) -> ndarray:
        """
        A positive value of the residual shows an X value that is higher than expected at that death time.

        """
        # TODO: the diff_against is gross
        # This uses Efron ties.

        X = X.values
        E = E.values
        T = T.values
        weights = weights.values

        n, d = X.shape

        if not np.any(E):
            # sometimes strata have no deaths. This means nothing is returned
            # in the below code.
            return zeros((n, d))

        # Init risk and tie sums to zero
        risk_phi, tie_phi = 0, 0
        risk_phi_x, tie_phi_x = zeros((1, d)), zeros((1, d))

        # Init number of ties and weights
        weight_count = 0.0
        tie_count = 0

        scores = weights * exp(dot(X, self.params_))

        diff_against = []

        schoenfeld_residuals = np.empty((0, d))

        # Iterate backwards to utilize recursive relationship
        for i in range(n - 1, -1, -1):
            # Doing it like this to preserve shape
            ti = T[i]
            ei = E[i]
            xi = X[i : i + 1]
            score = scores[i : i + 1]
            w = weights[i]

            # Calculate phi values
            phi_i = score
            phi_x_i = phi_i * xi

            # Calculate sums of Risk set
            risk_phi = risk_phi + phi_i
            risk_phi_x = risk_phi_x + phi_x_i

            # Calculate sums of Ties, if this is an event
            diff_against.append((xi, ei))
            if ei:

                tie_phi = tie_phi + phi_i
                tie_phi_x = tie_phi_x + phi_x_i

                # Keep track of count
                tie_count += 1  # aka death counts
                weight_count += w

            if i > 0 and T[i - 1] == ti:
                # There are more ties/members of the risk set
                continue
            elif tie_count == 0:
                for _ in diff_against:
                    schoenfeld_residuals = np.append(schoenfeld_residuals, zeros((1, d)), axis=0)
                diff_against = []
                continue

            # There was atleast one event and no more ties remain. Time to sum.
            weighted_mean = zeros((1, d))

            for l in range(tie_count):

                numer = risk_phi_x - l * tie_phi_x / tie_count
                denom = risk_phi - l * tie_phi / tie_count

                weighted_mean += numer / (denom * tie_count)

            for xi, ei in diff_against:
                schoenfeld_residuals = np.append(schoenfeld_residuals, ei * (xi - weighted_mean), axis=0)

            # reset tie values
            tie_count = 0
            weight_count = 0.0
            tie_phi = 0
            tie_phi_x = zeros((1, d))
            diff_against = []

        return schoenfeld_residuals[::-1]

    def _compute_delta_beta(
        self, X: DataFrame, T: Series, E: Series, weights: Series, index: Optional[Index] = None
    ) -> pd.DataFrame:
        """
        approximate change in betas as a result of excluding ith row. Good for finding outliers / specific
        subjects that influence the model disproportionately. Good advice: don't drop these outliers, model them.
        """
        score_residuals = self._compute_score(X, T, E, weights, index=index)

        d = X.shape[1]
        scaled_variance_matrix = self.variance_matrix_ * np.tile(self._norm_std.values, (d, 1)).T

        delta_betas = score_residuals.dot(scaled_variance_matrix)
        delta_betas.columns = self.params_.index

        return delta_betas

    def _compute_score(self, X: DataFrame, T: Series, E: Series, weights: Series, index: Optional[Index] = None) -> pd.DataFrame:

        _, d = X.shape

        if self.strata is not None:
            score_residuals = np.empty((0, d))

            for score_residuals_in_strata in self._partition_by_strata_and_apply(
                X, T, E, weights, None, self._compute_score_within_strata
            ):
                score_residuals = np.append(score_residuals, score_residuals_in_strata, axis=0)

        else:
            score_residuals = self._compute_score_within_strata(X, T, E, weights, None)

        return pd.DataFrame(score_residuals, columns=self.params_.index, index=index)

    def _compute_score_within_strata(self, X: DataFrame, _T: Series, E: Series, weights: Series, entries: None) -> ndarray:
        # https://www.stat.tamu.edu/~carroll/ftp/gk001.pdf
        # lin1989
        # https://www.ics.uci.edu/~dgillen/STAT255/Handouts/lecture10.pdf
        # Assumes X already sorted by T with strata
        # TODO: doesn't handle ties.
        # TODO: _T unused

        X = X.values
        E = E.values
        weights = weights.values

        n, d = X.shape

        # we already unnormalized the betas in `fit`, so we need normalize them again since X is
        # normalized.
        beta = self.params_.values * self._norm_std

        E = E.astype(int)
        score_residuals = zeros((n, d))

        phi_s = exp(dot(X, beta))

        # need to store these histories, as we access them often
        # this is a reverse cumulative sum. See original code in https://github.com/CamDavidsonPilon/lifelines/pull/496/files#diff-81ee0759dbae0770e1a02cf17f4cfbb1R431
        risk_phi_x_history = (X * (weights * phi_s)[:, None])[::-1].cumsum(0)[::-1]
        risk_phi_history = (weights * phi_s)[::-1].cumsum()[::-1][:, None]

        # Iterate forwards
        for i in range(0, n):

            xi = X[i : i + 1]
            phi_i = phi_s[i]

            score = -phi_i * (
                (E[: i + 1] * weights[: i + 1] / risk_phi_history[: i + 1].T).T  # this is constant-ish, and could be cached
                * (xi - risk_phi_x_history[: i + 1] / risk_phi_history[: i + 1])
            ).sum(0)

            if E[i]:
                score = score + (xi - risk_phi_x_history[i] / risk_phi_history[i])

            score_residuals[i, :] = score

        return score_residuals * weights[:, None]

    def _compute_confidence_intervals(self) -> pd.DataFrame:
        ci = 100 * (1 - self.alpha)
        z = utils.inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        hazards = self.params_.values
        return pd.DataFrame(
            np.c_[hazards - z * se, hazards + z * se],
            columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
            index=self.params_.index,
        )

    def _compute_standard_errors(
        self, X: Optional[DataFrame], T: Optional[Series], E: Optional[Series], weights: Optional[Series]
    ) -> pd.Series:
        if self.robust or self.cluster_col:
            se = np.sqrt(self._compute_sandwich_estimator(X, T, E, weights).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.values.diagonal())
        return pd.Series(se, name="se", index=self.params_.index)

    def _compute_sandwich_estimator(self, X: DataFrame, T: Series, E: Series, weights: Series) -> ndarray:
        delta_betas = self._compute_delta_beta(X, T, E, weights)

        if self.cluster_col:
            delta_betas = delta_betas.groupby(self._clusters).sum()

        sandwich_estimator = delta_betas.T.dot(delta_betas)

        return sandwich_estimator.values

    def _compute_z_values(self) -> Series:
        return self.params_ / self.standard_errors_

    def _compute_p_values(self) -> ndarray:
        #  Survival Analysis by John P. Klein and Melvin L. Moeschberger, Second Edition, page 256
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    @property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
        """
        ci = 100 * (1 - self.alpha)
        z = utils.inv_normal_cdf(1 - self.alpha / 2)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = self.hazard_ratios_
            df["se(coef)"] = self.standard_errors_
            df["coef lower %g%%" % ci] = self.confidence_intervals_["%g%% lower-bound" % ci]
            df["coef upper %g%%" % ci] = self.confidence_intervals_["%g%% upper-bound" % ci]
            df["exp(coef) lower %g%%" % ci] = self.hazard_ratios_ * exp(-z * self.standard_errors_)
            df["exp(coef) upper %g%%" % ci] = self.hazard_ratios_ * exp(z * self.standard_errors_)
            df["cmp to"] = np.zeros_like(self.params_)
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -utils.quiet_log2(df["p"])
            return df

    def _trivial_log_likelihood(self):
        df = pd.DataFrame({"T": self.durations, "E": self.event_observed, "W": self.weights})
        if self.entry_col is not None:
            df["entry"] = self.entries
            trivial_model = self.__class__().fit_right_censoring(df, "E", weights_col="W", entry_col="entry")
        else:
            trivial_model = self.__class__().fit_right_censoring(df, "E", weights_col="W")
        return trivial_model.log_likelihood_

    def log_likelihood_ratio_test(self) -> StatisticalResult:
        """
        This function computes the likelihood ratio test for the Cox model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.
        """
        if hasattr(self, "_ll_null_"):
            ll_null = self._ll_null_
        else:
            ll_null = self._trivial_log_likelihood()
        ll_alt = self.log_likelihood_
        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.params_.shape[0]
        p_value = _chisq_test_p_value(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value,
            test_stat,
            test_name="log-likelihood ratio test",
            null_distribution="chi squared",
            degrees_freedom=degrees_freedom,
        )

    def predict_partial_hazard(self, X: Union[ndarray, DataFrame]) -> pd.Series:
        r"""
        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`\exp{(x - mean(x_{train}))'\beta}`

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Notes
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        return exp(self.predict_log_partial_hazard(X))

    def predict_log_partial_hazard(self, X: Union[ndarray, DataFrame]) -> pd.Series:
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - \text{mean}(x_{\text{train}})) \beta`


        Parameters
        ----------
        X:  numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Notes
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        hazard_names = self.params_.index

        if isinstance(X, pd.Series) and ((X.shape[0] == len(hazard_names) + 2) or (X.shape[0] == len(hazard_names))):
            X = X.to_frame().T.infer_objects()
            return self.predict_log_partial_hazard(X)
        elif isinstance(X, pd.Series):
            assert len(hazard_names) == 1, "Series not the correct argument"
            X = X.to_frame().T.infer_objects()
            return self.predict_log_partial_hazard(X)

        index = utils._get_index(X)

        if isinstance(X, pd.DataFrame):

            X = self.regressors.transform_df(X)["beta_"]
            X = X.values

        X = X.astype(float)
        X = utils.normalize(X, self._norm_mean.values, 1)
        return pd.Series(dot(X, self.params_), index=index)

    def predict_cumulative_hazard(
        self,
        X: Union[Series, DataFrame],
        times: Optional[Union[ndarray, List[float]]] = None,
        conditional_after: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
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
        conditional_after: iterable, optional
            Must be equal is size to X.shape[0] (denoted ``n`` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents :math:`s` in
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. reset back to starting at 0.

        """
        if isinstance(X, pd.Series):
            return self.predict_cumulative_hazard(
                X.to_frame().T.infer_objects(), times=times, conditional_after=conditional_after
            )

        n = X.shape[0]

        if times is not None:
            times = np.atleast_1d(times).astype(float)
        else:
            times = self.timeline
        if conditional_after is not None:
            conditional_after = utils._to_1d_array(conditional_after).reshape(n, 1)

        if self.strata:
            X = X.copy()
            cumulative_hazard_ = pd.DataFrame()
            if conditional_after is not None:
                X["_conditional_after"] = conditional_after

            for stratum, stratified_X in X.groupby(self.strata):
                try:
                    strata_c_0 = self.baseline_cumulative_hazard_[[stratum]]
                except KeyError:
                    raise exceptions.StatError(
                        dedent(
                            """The stratum %s was not found in the original training data. For example, try
                            the following on the original dataset, df: `df.groupby(%s).size()`. Expected is that %s is not present in the output."""
                            % (stratum, self.strata, stratum)
                        )
                    )
                col = utils._get_index(stratified_X)
                v = self.predict_partial_hazard(stratified_X)
                times_ = times
                n_ = stratified_X.shape[0]
                if conditional_after is not None:
                    conditional_after_ = stratified_X.pop("_conditional_after").values[:, None]
                    times_to_evaluate_at = np.tile(times_, (n_, 1)) + conditional_after_

                    c_0_ = utils.interpolate_at_times(strata_c_0, times_to_evaluate_at)
                    c_0_conditional_after = utils.interpolate_at_times(strata_c_0, conditional_after_)
                    c_0_ = np.clip((c_0_ - c_0_conditional_after).T, 0, np.inf)

                else:
                    times_to_evaluate_at = np.tile(times_, (n_, 1))
                    c_0_ = utils.interpolate_at_times(strata_c_0, times_to_evaluate_at).T

                cumulative_hazard_ = cumulative_hazard_.merge(
                    pd.DataFrame(c_0_ * v.values, columns=col, index=times_), how="outer", right_index=True, left_index=True
                )
        else:

            v = self.predict_partial_hazard(X)
            col = utils._get_index(v)
            times_ = times

            if conditional_after is not None:
                times_to_evaluate_at = np.tile(times_, (n, 1)) + conditional_after

                c_0 = utils.interpolate_at_times(self.baseline_cumulative_hazard_, times_to_evaluate_at)
                c_0_conditional_after = utils.interpolate_at_times(self.baseline_cumulative_hazard_, conditional_after)
                c_0 = np.clip((c_0 - c_0_conditional_after).T, 0, np.inf)

            else:
                times_to_evaluate_at = np.tile(times_, (n, 1))
                c_0 = utils.interpolate_at_times(self.baseline_cumulative_hazard_, times_to_evaluate_at).T

            cumulative_hazard_ = pd.DataFrame(c_0 * v.values, columns=col, index=times_)

        return cumulative_hazard_

    def predict_hazard(*args, **kwargs):
        raise NotImplementedError(
            "This can't be reliably computed for the Cox proportional hazard model with Breslow baseline hazard."
        )

    def predict_survival_function(
        self,
        X: Union[Series, DataFrame],
        times: Optional[Union[List[float], ndarray]] = None,
        conditional_after: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Predict the survival function for individuals, given their covariates. This assumes that the individual
        just entered the study (that is, we do not condition on how long they have already lived for.)

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
        conditional_after: iterable, optional
            Must be equal is size to X.shape[0] (denoted ``n`` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents :math:`s` in
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.

        """
        return exp(-self.predict_cumulative_hazard(X, times=times, conditional_after=conditional_after))

    def predict_percentile(self, X: DataFrame, p: float = 0.5, conditional_after: Optional[ndarray] = None) -> pd.Series:
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
        p: float, optional (default=0.5)
            the percentile, must be between 0 and 1.
        conditional_after: iterable, optional
            Must be equal is size to X.shape[0] (denoted ``n`` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents :math:`s` in
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.

        See Also
        --------
        predict_median

        """
        subjects = utils._get_index(X)
        return utils.qth_survival_times(
            p, self.predict_survival_function(X, conditional_after=conditional_after)[subjects]
        ).T.squeeze()

    def predict_median(self, X: DataFrame, conditional_after: Optional[ndarray] = None) -> pd.Series:
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        conditional_after: iterable, optional
            Must be equal is size to X.shape[0] (denoted ``n`` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents :math:`s` in
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.


        See Also
        --------
        predict_percentile

        """
        return self.predict_percentile(X, 0.5, conditional_after=conditional_after)

    def predict_expectation(self, X: DataFrame, conditional_after: Optional[ndarray] = None) -> pd.Series:
        r"""
        Compute the expected lifetime, :math:`E[T]`, using covariates X. This algorithm to compute the expectation is
        to use the fact that :math:`E[T] = \int_0^\inf P(T > t) dt = \int_0^\inf S(t) dt`. To compute the integral, we use the trapezoidal rule to approximate the integral.

        Caution
        --------
        If the survival function doesn't converge to 0, then the expectation is really infinity and the returned
        values are meaningless/too large. In that case, using ``predict_median`` or ``predict_percentile`` would be better.

        Parameters
        ----------

        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        conditional_after: iterable, optional
            Must be equal is size to X.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents :math:`s` in
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.

        Notes
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        See Also
        --------
        predict_median
        predict_percentile

        """
        subjects = utils._get_index(X)
        v = self.predict_survival_function(X, conditional_after=conditional_after)[subjects]
        return pd.Series(trapezoid(v.values.T, v.index), index=subjects)

    def _compute_baseline_hazard(self, partial_hazards: DataFrame, name: Any) -> pd.DataFrame:
        # https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
        ind_hazards = partial_hazards.copy()
        ind_hazards["P"] *= ind_hazards["W"]
        ind_hazards["E"] *= ind_hazards["W"]
        ind_hazards_summed_over_durations = ind_hazards.groupby("T")[["P", "E"]].sum()
        ind_hazards_summed_over_durations["P"] = ind_hazards_summed_over_durations["P"].loc[::-1].cumsum()
        baseline_hazard = pd.DataFrame(
            ind_hazards_summed_over_durations["E"] / ind_hazards_summed_over_durations["P"], columns=[name]
        )
        baseline_hazard.index.name = None
        return baseline_hazard

    def _compute_baseline_hazards(self, predicted_partial_hazards_) -> pd.DataFrame:
        if self.strata:

            index = self.durations.unique()
            baseline_hazards_ = pd.DataFrame(index=index).sort_index()

            for name, stratum_predicted_partial_hazards_ in predicted_partial_hazards_.groupby(self.strata):
                baseline_hazards_ = baseline_hazards_.merge(
                    self._compute_baseline_hazard(stratum_predicted_partial_hazards_, name),
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            return baseline_hazards_.fillna(0)

        return self._compute_baseline_hazard(predicted_partial_hazards_, name="baseline hazard")

    def _compute_baseline_cumulative_hazard(self, baseline_hazard_) -> DataFrame:
        cumulative = baseline_hazard_.cumsum()
        if not self.strata:
            cumulative = cumulative.rename(columns={"baseline hazard": "baseline cumulative hazard"})
        return cumulative

    def _compute_baseline_survival(self) -> pd.DataFrame:
        """
        Importantly, this agrees with what the KaplanMeierFitter produces. Ex:

        Example
        -------
        .. code:: python

            from lifelines.datasets import load_rossi
            from lifelines import CoxPHFitter, KaplanMeierFitter
            rossi = load_rossi()

            kmf = KaplanMeierFitter()
            kmf.fit(rossi['week'], rossi['arrest'])
            rossi2 = rossi[['week', 'arrest']].copy()
            rossi2['var1'] = np.random.randn(432)

            cph = CoxPHFitter()
            cph.fit(rossi2, 'week', 'arrest')
            ax = cph.baseline_survival_.plot()
            kmf.plot(ax=ax)
        """
        survival_df = exp(-self.baseline_cumulative_hazard_)
        if not self.strata:
            survival_df = survival_df.rename(columns={"baseline cumulative hazard": "baseline survival"})
        return survival_df

    def plot(self, columns=None, hazard_ratios=False, ax=None, **errorbar_kwargs):
        """
        Produces a visual representation of the coefficients (i.e. log hazard ratios), including their standard errors and magnitudes.

        Parameters
        ----------
        columns : list, optional
            specify a subset of the columns to plot
        hazard_ratios: bool, optional
            by default, ``plot`` will present the log-hazard ratios (the coefficients). However, by turning this flag to True, the hazard ratios are presented instead.
        errorbar_kwargs:
            pass in additional plotting commands to matplotlib errorbar command

        Examples
        ---------
        .. code:: python

            from lifelines import datasets, CoxPHFitter
            rossi = datasets.load_rossi()
            cph = CoxPHFitter().fit(rossi, 'week', 'arrest')
            cph.plot(hazard_ratios=True)

        Returns
        -------
        ax: matplotlib axis
            the matplotlib axis that be edited.

        """
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()

        errorbar_kwargs.setdefault("c", "k")
        errorbar_kwargs.setdefault("fmt", "s")
        errorbar_kwargs.setdefault("markerfacecolor", "white")
        errorbar_kwargs.setdefault("markeredgewidth", 1.25)
        errorbar_kwargs.setdefault("elinewidth", 1.25)
        errorbar_kwargs.setdefault("capsize", 3)

        z = utils.inv_normal_cdf(1 - self.alpha / 2)
        user_supplied_columns = True

        if columns is None:
            user_supplied_columns = False
            columns = self.params_.index

        yaxis_locations = list(range(len(columns)))
        log_hazards = self.params_.loc[columns].values.copy()

        order = list(range(len(columns) - 1, -1, -1)) if user_supplied_columns else np.argsort(log_hazards)

        if hazard_ratios:
            exp_log_hazards = exp(log_hazards)
            upper_errors = exp_log_hazards * (exp(z * self.standard_errors_[columns].values) - 1)
            lower_errors = exp_log_hazards * (1 - exp(-z * self.standard_errors_[columns].values))
            ax.errorbar(
                exp_log_hazards[order],
                yaxis_locations,
                xerr=np.vstack([lower_errors[order], upper_errors[order]]),
                **errorbar_kwargs,
            )
            ax.set_xlabel("HR (%g%% CI)" % ((1 - self.alpha) * 100))
        else:
            symmetric_errors = z * self.standard_errors_[columns].values
            ax.errorbar(log_hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
            ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - self.alpha) * 100))

        best_ylim = ax.get_ylim()
        ax.vlines(1 if hazard_ratios else 0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
        ax.set_ylim(best_ylim)

        tick_labels = [columns[i] for i in order]

        ax.set_yticks(yaxis_locations)
        ax.set_yticklabels(tick_labels)

        return ax

    def score(self, df: pd.DataFrame, scoring_method: str = "log_likelihood") -> float:
        """
        Score the data in df on the fitted model. With default scoring method, returns
        the *average partial log-likelihood*.

        Parameters
        ----------
        df: DataFrame
            the dataframe with duration col, event col, etc.
        scoring_method: str
            one of {'log_likelihood', 'concordance_index'}
            log_likelihood: returns the average unpenalized partial log-likelihood.
            concordance_index: returns the concordance-index

        Examples
        ---------

        .. code:: python

            from lifelines import CoxPHFitter
            from lifelines.datasets import load_rossi

            rossi_train = load_rossi().loc[:400]
            rossi_test = load_rossi().loc[400:]
            cph = CoxPHFitter().fit(rossi_train, 'week', 'arrest')

            cph.score(rossi_train)
            cph.score(rossi_test)

        """
        df = df.copy()

        if self.strata:
            df = df.set_index(self.strata)

        df = df.sort_values([self.duration_col, self.event_col])
        T = df.pop(self.duration_col).astype(float)
        E = df.pop(self.event_col).astype(bool)
        W = df.pop(self.weights_col) if self.weights_col else pd.Series(np.ones_like(E), index=T.index)
        entries = df.pop(self.entry_col) if self.entry_col else None

        if scoring_method == "log_likelihood":

            df = self.regressors.transform_df(df)["beta_"]
            df = utils.normalize(df, self._norm_mean.values, 1.0)

            get_gradients = self._choose_gradient_calculator(T, df, entries)
            optimal_beta = self.params_.values

            if self.strata is None:
                *_, ll_ = get_gradients(df, T, E, W, entries, optimal_beta)
            else:
                ll_ = sum(r[-1] for r in self._partition_by_strata_and_apply(df, T, E, W, entries, get_gradients, optimal_beta))
            return ll_ / df.shape[0]

        elif scoring_method == "concordance_index":
            predictions = -self.predict_partial_hazard(df)
            # TODO: move strata logic into concordance_index.
            return concordance_index(T, predictions, event_observed=E)

        else:
            raise NotImplementedError()

    @property
    def concordance_index_(self) -> float:
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the ROC AUC
        to survival data, including censoring.

        For this purpose, the ``concordance_index_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset.

        References
        ----------
        https://stats.stackexchange.com/questions/133817/stratified-concordance-index-survivalsurvconcordance

        """
        # pylint: disable=access-member-before-definition

        if not hasattr(self, "_concordance_index_"):
            if self.strata:
                # https://stats.stackexchange.com/questions/133817/stratified-concordance-index-survivalsurvconcordance
                num_correct, num_tied, num_pairs = 0, 0, 0
                for _, _df in self._predicted_partial_hazards_.groupby(self.strata):
                    if _df.shape[0] == 1:
                        continue
                    _num_correct, _num_tied, _num_pairs = _concordance_summary_statistics(
                        _df["T"].values, -_df["P"].values, _df["E"].values
                    )
                    num_correct += _num_correct
                    num_tied += _num_tied
                    num_pairs += _num_pairs
            else:
                df = self._predicted_partial_hazards_
                num_correct, num_tied, num_pairs = _concordance_summary_statistics(
                    df["T"].values, -df["P"].values, df["E"].values
                )

            self._concordance_index_ = _concordance_ratio(num_correct, num_tied, num_pairs)
            return self.concordance_index_
        return self._concordance_index_

    @property
    def AIC_(self):
        raise exceptions.StatError(
            "Since the model is semi-parametric (and not fully-parametric), the AIC does not exist. You probably want the `.AIC_partial_` property instead."
        )


class ParametricCoxModelFitter(ParametricRegressionFitter, ProportionalHazardMixin):

    _KNOWN_MODEL = True
    cluster_col = None

    def _cumulative_hazard(self, params, T, Xs):
        if self.strata is not None:
            return self._cumulative_hazard_with_strata(params, T, Xs)
        else:
            return self._cumulative_hazard_sans_strata(params, T, Xs)

    @property
    def baseline_hazard_(self):
        return self.baseline_hazard_at_times(self.timeline)

    @property
    def baseline_survival_(self):
        return self.baseline_survival_at_times(self.timeline)

    @property
    def baseline_cumulative_hazard_(self):
        return self.baseline_cumulative_hazard_at_times(self.timeline)

    def baseline_hazard_at_times(self, times=None):
        """
        Predict the baseline hazard at times (Defaults to observed durations)
        """
        times = utils.coalesce(times, self.timeline)
        if self.strata is not None:
            v = self.predict_hazard(self._central_values.reset_index(), times=times)
            v.columns = self._central_values.index.values
        else:
            v = self.predict_hazard(self._central_values, times=times)
            v.columns = ["baseline hazard"]
        return v

    def baseline_survival_at_times(self, times: Iterable = None):
        """
        Predict the baseline survival at times (Defaults to observed durations)
        """
        times = utils.coalesce(times, self.timeline)
        if self.strata is not None:
            v = self.predict_survival_function(self._central_values.reset_index(), times=times)
            v.columns = self._central_values.index.values
        else:
            v = self.predict_survival_function(self._central_values, times=times)
            v.columns = ["baseline survival"]
        return v

    def baseline_cumulative_hazard_at_times(self, times: Iterable = None):
        """
        Predict the baseline cumulative hazard at times (Defaults to observed durations)
        """
        times = utils.coalesce(times, self.timeline)
        if self.strata is not None:
            v = self.predict_cumulative_hazard(self._central_values.reset_index(), times=times)
            v.columns = self._central_values.index.values
        else:
            v = self.predict_cumulative_hazard(self._central_values, times=times)
            v.columns = ["baseline cumulative hazard"]
        return v

    def predict_cumulative_hazard(self, df, *, times: Iterable = None, conditional_after: Iterable = None):
        """
        Predict the cumulative hazard for individuals, given their covariates.

        Parameters
        ----------

        df: DataFrame
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order.
        times: iterable, optional
            an iterable (array, list, series) of increasing times to predict the cumulative hazard at. Default
            is the set of all durations in the training dataset (observed and unobserved).
        conditional_after: iterable, optional
            Must be equal is size to (df.shape[0],) (`n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.

        Returns
        -------
        DataFrame
            the cumulative hazards of individuals over the timeline

        """
        if isinstance(df, pd.Series):
            df = df.to_frame().T.infer_objects()

        df = df.copy()
        df.index.name = None

        if self.strata is not None:
            df = df.reset_index().set_index(self.strata)

            cumulative_hazard = pd.DataFrame()
            if conditional_after is not None:
                # need to pass this into the groupby
                df["conditional_after_"] = conditional_after

            for stratum, stratified_X in df.groupby(self.strata):

                if conditional_after is not None:
                    conditional_after_ = stratified_X.pop("conditional_after_")
                else:
                    conditional_after_ = None

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    cumulative_hazard_ = super(ParametricCoxModelFitter, self).predict_cumulative_hazard(
                        stratified_X, times=times, conditional_after=conditional_after_
                    )
                cumulative_hazard_.columns = stratified_X["index"]
                cumulative_hazard = cumulative_hazard.merge(cumulative_hazard_, how="outer", right_index=True, left_index=True)

            return cumulative_hazard

        else:
            return super(ParametricCoxModelFitter, self).predict_cumulative_hazard(
                df, times=times, conditional_after=conditional_after
            )

    def predict_hazard(self, df, *, conditional_after=None, times=None):
        """
        Predict the hazard for individuals, given their covariates.

        Parameters
        ----------

        df: DataFrame
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order.
        times: iterable, optional
            an iterable (array, list, series) of increasing times to predict the cumulative hazard at. Default
            is the set of all durations in the training dataset (observed and unobserved).
        conditional_after:
            Not implemented yet.

        Returns
        -------
        DataFrame
            the hazards of individuals over the timeline

        """
        if isinstance(df, pd.Series):
            df = df.to_frame().T.infer_objects()

        df = df.copy()

        if self.strata is not None:
            df = df.reset_index().set_index(self.strata)

            cumulative_hazard = pd.DataFrame()
            if conditional_after is not None:
                # need to pass this into the groupby
                df["conditional_after_"] = conditional_after

            for stratum, stratified_X in df.groupby(self.strata):

                if conditional_after is not None:
                    conditional_after_ = stratified_X.pop("conditional_after_")
                else:
                    conditional_after_ = None

                cumulative_hazard_ = super(ParametricCoxModelFitter, self).predict_hazard(
                    stratified_X, times=times, conditional_after=conditional_after_
                )
                cumulative_hazard_.columns = stratified_X["index"]
                cumulative_hazard = cumulative_hazard.merge(cumulative_hazard_, how="outer", right_index=True, left_index=True)

            return cumulative_hazard

        else:
            return super(ParametricCoxModelFitter, self).predict_hazard(df, times=times, conditional_after=conditional_after)

    @property
    def AIC_partial_(self):
        raise exceptions.StatError(
            "Since the spline model is fully parametric (and not semi-parametric), the partial AIC does not exist. You probably want the `.AIC_` property instead."
        )


class ParametricSplinePHFitter(ParametricCoxModelFitter, SplineFitterMixin):
    r"""
    Proportional hazard model with cubic splines model for the baseline hazard.

    .. math::  H(t|x) = H_0(t) \exp(x' \beta)

    where

    .. math:: H_0(t) = \exp{\left( \phi_0 + \phi_1\log{t} + \sum_{j=2}^N \phi_j v_j(\log{t})\right)}

    where :math:`v_j` are our cubic basis functions at predetermined knots, and :math:`H_0` is the cumulative baseline hazard. See references for exact definition.

    References
    ------------
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards and proportional-odds models for censored survival data, with application to prognostic modelling and estimation of treatment effects. Statistics in Medicine, 21(15), 2175–2197. doi:10.1002/sim.1203

    Note
    -------
    This is a "hidden" class that is invoked when using ``baseline_estimation_method="spline"``. You probably want to use ``CoxPHFitter``, not this.
    """

    _scipy_fit_method = "SLSQP"
    _scipy_fit_options = {"maxiter": 1000, "iprint": 100}

    _FAST_MEDIAN_PREDICT = False
    fit_intercept = True

    def __init__(self, strata, strata_values, n_baseline_knots=1, knots=None, *args, **kwargs):
        self.strata = utils._to_list_or_singleton(strata)
        self.strata_values = strata_values

        assert (
            n_baseline_knots is not None and n_baseline_knots > 1
        ), "n_baseline_knots should be greater than 1. Set in class instantiation"

        self.n_baseline_knots = n_baseline_knots
        self.knots = knots
        super(ParametricSplinePHFitter, self).__init__(*args, **kwargs)

    @staticmethod
    def _strata_labeler(stratum, i):
        try:
            return "s%s_phi%d_" % (tuple(str(s) for s in stratum), i)
        except:
            # singleton
            return "s%s_phi%d_" % (stratum, i)

    @property
    def _fitted_parameter_names(self):
        if self.strata is not None:
            names = ["beta_"]
            for stratum in self.strata_values:
                names += [self._strata_labeler(stratum, i) for i in range(1, self.n_baseline_knots + 1)]
            return names
        else:
            return ["beta_"] + ["phi%d_" % i for i in range(1, self.n_baseline_knots + 1)]

    def _set_knots(self, T, E):
        self.knots = np.percentile(T[E.astype(bool).values], np.linspace(5, 95, self.n_baseline_knots + 1))
        return

    def _pre_fit_model(self, Ts, E, df):
        if self.knots is not None:
            return
        if E.sum() > 4:
            self._set_knots(utils.coalesce(*Ts), E)
        else:
            # very few observations
            self._set_knots(utils.coalesce(*Ts), pd.Series(np.ones_like(E)))

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        #  Some non-zero initial points. This is important as it nudges the model slightly away from the degenerate all-zeros model. Try setting it to 0, and watch the model fail to converge.
        if self.strata is not None:
            params = {"beta_": np.zeros(len(Xs["beta_"].columns))}
            for stratum in self.strata_values:
                params.update({self._strata_labeler(stratum, i): np.array([0.001]) for i in range(1, self.n_baseline_knots + 1)})

            return params

        else:
            return {
                **{"beta_": np.zeros(len(Xs["beta_"].columns))},
                **{"phi%d_" % i: np.array([0.01]) for i in range(1, self.n_baseline_knots + 1)},
            }

    def _cumulative_hazard_with_strata(self, params, T, Xs):
        lT = anp.log(T)
        output = []

        # hack for iterating over stratified T
        start, stop = 0, 0

        # I can assume Xs is sorted by strata values
        for stratum, Xs_ in Xs.groupby(self.strata):
            stop = stop + Xs_.size

            if T.ndim > 1:
                lT_ = lT[:, start:stop]
            else:
                lT_ = lT[start:stop]

            H_ = safe_exp(anp.dot(Xs_["beta_"], params["beta_"]) + params[self._strata_labeler(stratum, 1)] * lT_)

            for i in range(2, self.n_baseline_knots + 1):
                H_ = H_ * safe_exp(
                    params[self._strata_labeler(stratum, i)]
                    * self.basis(lT_, anp.log(self.knots[i - 1]), anp.log(self.knots[0]), anp.log(self.knots[-1]))
                )

            output.append(H_)
            start = stop

        return anp.hstack(output) if output else anp.array([])

    def _cumulative_hazard_sans_strata(self, params, T, Xs):
        lT = anp.log(T)

        H = safe_exp(anp.dot(Xs["beta_"], params["beta_"]) + params["phi1_"] * lT)

        for i in range(2, self.n_baseline_knots + 1):
            H = H * safe_exp(
                params["phi%d_" % i] * self.basis(lT, anp.log(self.knots[i - 1]), anp.log(self.knots[0]), anp.log(self.knots[-1]))
            )
        return H


class ParametricPiecewiseBaselinePHFitter(ParametricCoxModelFitter, ProportionalHazardMixin):
    r"""
    Proportional hazard model with piecewise constant model for the baseline hazard.

    .. math::  h(t|x) = h_0(t) \exp(x' \beta)

    where

    .. math::  h_0(t) = \begin{cases}
                        exp{\beta \cdot \text{center}(x)}  & \text{if $t \le \tau_0$} \\
                        exp{\beta \cdot \text{center}(x)} \cdot lambda_1 & \text{if $\tau_0 < t \le \tau_1$} \\
                        exp{\beta \cdot \text{center}(x)} \cdot lambda_2 & \text{if $\tau_1 < t \le \tau_2$} \\
                        ...
                      \end{cases}

    Note
    -------
    This is a "hidden" class that is invoked when using ``baseline_estimation_method="piecewise"``. You probably want to use ``CoxPHFitter``, not this.
    """

    _KNOWN_MODEL = True
    _FAST_MEDIAN_PREDICT = False
    fit_intercept = True

    cluster_col = None

    def __init__(self, strata, strata_values, breakpoints, *args, **kwargs):
        self.strata = utils._to_list_or_singleton(strata)
        self.strata_values = strata_values

        assert (
            breakpoints is not None and len(breakpoints) > 0
        ), "breakpoints should be greater than 0. Set in class instantiation"

        self.breakpoints = breakpoints
        self.n_breakpoints = len(breakpoints)
        super(ParametricPiecewiseBaselinePHFitter, self).__init__(*args, **kwargs)

    @staticmethod
    def _strata_labeler(stratum, i):
        try:
            return "s%s_lambda%d_" % (tuple(str(s) for s in stratum), i)
        except:
            # singleton
            return "s%s_lambda%d_" % (stratum, i)

    @property
    def _fitted_parameter_names(self):
        if self.strata is not None:
            names = ["beta_"]
            for stratum in self.strata_values:
                names += [self._strata_labeler(stratum, i) for i in range(2, self.n_breakpoints + 2)]
            return names
        else:
            return ["beta_"] + ["log_lambda%d_" % i for i in range(2, self.n_breakpoints + 2)]

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        #  Some non-zero initial points. This is important as it nudges the model slightly away from the degenerate all-zeros model. Try setting it to 0, and watch the model fail to converge.
        if self.strata is not None:
            params = {"beta_": np.zeros(len(Xs["beta_"].columns))}
            for stratum in self.strata_values:
                params.update({self._strata_labeler(stratum, i): np.array([0.001]) for i in range(2, self.n_breakpoints + 2)})

            return params

        else:
            return {
                **{"beta_": np.zeros(len(Xs["beta_"].columns))},
                **{"log_lambda%d_" % i: np.array([0.001]) for i in range(2, self.n_breakpoints + 2)},
            }

    def _cumulative_hazard_with_strata(self, params, T, Xs):
        output = []

        # hack for iterating over stratified T
        start, stop = 0, 0

        # I can assume Xs is sorted by strata values
        for stratum, Xs_ in Xs.groupby(self.strata):
            stop = stop + Xs_.size

            if T.ndim > 1:
                T_ = T[:, start:stop]
            else:
                T_ = T[start:stop]

            partial_hazard = safe_exp(anp.dot(Xs_["beta_"], params["beta_"]))
            n = T_.shape[0]
            T_ = T_.reshape((n, 1))
            bps = anp.append(self.breakpoints, [anp.inf])
            M = anp.minimum(anp.tile(bps, (n, 1)), T_)
            M = anp.hstack([M[:, tuple([0])], anp.diff(M, axis=1)])
            log_lambdas_ = anp.array(
                [0] + [params[self._strata_labeler(stratum, i)][0] for i in range(2, self.n_breakpoints + 2)]
            )
            H_ = partial_hazard * (M * anp.exp(log_lambdas_).T).sum(1)

            output.append(H_)
            start = stop

        return anp.hstack(output) if output else anp.array([])

    def _cumulative_hazard_sans_strata(self, params, T, Xs):
        partial_hazard = safe_exp(anp.dot(Xs["beta_"], params["beta_"]))
        n = T.shape[0]
        T = T.reshape((n, 1))
        bps = anp.append(self.breakpoints, [anp.inf])
        M = anp.minimum(anp.tile(bps, (n, 1)), T)
        M = anp.hstack([M[:, tuple([0])], anp.diff(M, axis=1)])
        log_lambdas_ = anp.array([0.0] + [params[param][0] for param in self._fitted_parameter_names if param != "beta_"])
        return partial_hazard * (M * anp.exp(log_lambdas_).T).sum(1)

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
        df = df.copy()
        if isinstance(df, pd.Series):
            return self.predict_cumulative_hazard(df.to_frame().T)

        if conditional_after is not None:
            raise NotImplementedError()

        times = np.atleast_1d(utils.coalesce(times, self.timeline)).astype(float)
        n = times.shape[0]
        times = times.reshape((n, 1))

        bp = np.append(self.breakpoints, [np.inf])

        M = np.minimum(np.tile(bp, (n, 1)), times)
        M = np.hstack([M[:, tuple([0])], np.diff(M, axis=1)])

        if self.strata is not None:
            df = df.reset_index().set_index(self.strata)

            cumulative_hazard = pd.DataFrame()

            for stratum, stratified_X in df.groupby(self.strata):
                log_lambdas_ = anp.array(
                    [0] + [self.params_.loc[self._strata_labeler(stratum, i)].iloc[0] for i in range(2, self.n_breakpoints + 2)]
                )
                lambdas_ = np.exp(log_lambdas_)

                Xs_ = self.regressors.transform_df(stratified_X)
                partial_hazard = np.exp(np.dot(Xs_["beta_"], self.params_["beta_"]))

                cumulative_hazard_ = pd.DataFrame(np.outer(np.dot(M, lambdas_), partial_hazard), index=times[:, 0])
                cumulative_hazard_.columns = stratified_X["index"]
                cumulative_hazard = cumulative_hazard.merge(cumulative_hazard_, how="outer", right_index=True, left_index=True)

            return cumulative_hazard

        else:
            log_lambdas_ = np.array(
                [0] + [self.params_.loc[param].iloc[0] for param in self._fitted_parameter_names if param != "beta_"]
            )
            lambdas_ = np.exp(log_lambdas_)

            Xs = self.regressors.transform_df(df)
            partial_hazard = np.exp(np.dot(Xs["beta_"], self.params_["beta_"]))
            return pd.DataFrame(np.outer(np.dot(M, lambdas_), partial_hazard), columns=utils._get_index(df), index=times[:, 0])


class _BatchVsSingle:

    BATCH = "batch"
    SINGLE = "single"

    def decide(self, batch_mode: Optional[bool], n_unique: int, n_total: int, n_vars: int) -> str:
        log_frac_dups = np.log(n_unique / n_total)
        if batch_mode or (
            # https://github.com/CamDavidsonPilon/lifelines/issues/591 for original issue.
            # new values from from perf/batch_vs_single script.
            (batch_mode is None)
            and (
                1.537_271e00
                + n_total * 4.771_387e-06
                + log_frac_dups * 2.610_877e-01
                + n_total * log_frac_dups * -3.830_987e-11
                + log_frac_dups**2 * 1.389_890e-02
                + n_total**2 * 3.129_870e-14
                + n_vars * 3.196_517e-03
                + n_vars * n_total * -7.356_722e-07
            )
            < 1
        ):
            return self.BATCH
        return self.SINGLE
