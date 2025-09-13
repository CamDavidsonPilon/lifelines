# -*- coding: utf-8 -*-
import warnings
from datetime import datetime
import time

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.integrate import trapezoid

from lifelines.fitters import RegressionFitter
from lifelines.utils.printer import Printer
from lifelines.exceptions import StatisticalWarning, ConvergenceWarning
from lifelines import utils
from lifelines.utils import (
    _get_index,
    inv_normal_cdf,
    epanechnikov_kernel,
    ridge_regression as lr,
    qth_survival_times,
    check_for_numeric_dtypes_or_raise,
    concordance_index,
    check_nans_or_infs,
    normalize,
    string_rjustify,
    _to_list,
    format_floats,
    format_p_value,
    format_exp_floats,
    survival_table_from_events,
    CensoringType,
)


class AalenAdditiveFitter(RegressionFitter):

    r"""
    This class fits the regression model:

    .. math::  h(t|x)  = b_0(t) + b_1(t) x_1 + ... + b_N(t) x_N

    that is, the hazard rate is a linear function of the covariates with time-varying coefficients.
    This implementation assumes non-time-varying covariates, see ``TODO: name``

    Note
    -----

    This class was rewritten in lifelines 0.17.0 to focus solely on static datasets.
    There is no guarantee of backwards compatibility.

    Parameters
    -----------
    fit_intercept: bool, optional (default: True)
      If False, do not attach an intercept (column of ones) to the covariate matrix. The
      intercept, :math:`b_0(t)` acts as a baseline hazard.
    alpha: float, optional (default=0.05)
      the level in the confidence intervals.
    coef_penalizer: float, optional (default: 0)
      Attach a L2 penalizer to the size of the coefficients during regression. This improves
      stability of the estimates and controls for high correlation between covariates.
      For example, this shrinks the magnitude of :math:`c_{i,t}`.
    smoothing_penalizer: float, optional (default: 0)
      Attach a L2 penalizer to difference between adjacent (over time) coefficients. For
      example, this shrinks the magnitude of :math:`c_{i,t} - c_{i,t+1}`.

    Attributes
    ----------
    cumulative_hazards_ : DataFrame
        The estimated cumulative hazard
    hazards_ : DataFrame
        The estimated hazards
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the cumulative hazard
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    weights: array
        The event_observed variable provided
    """
    _KNOWN_MODEL = True

    def __init__(self, fit_intercept=True, alpha=0.05, coef_penalizer=0.0, smoothing_penalizer=0.0):
        super(AalenAdditiveFitter, self).__init__(alpha=alpha)
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.coef_penalizer = coef_penalizer
        self.smoothing_penalizer = smoothing_penalizer

        if coef_penalizer < 0 or smoothing_penalizer < 0:
            raise ValueError("penalizer parameters must be >= 0.")

    @CensoringType.right_censoring
    def fit(self, df, duration_col, event_col=None, weights_col=None, show_progress=False, formula: str = None):
        """
        Parameters
        ----------
        Fit the Aalen Additive model to a dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights).
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
            This can be used for sampling weights.

        show_progress: bool, optional (default=False)
            Since the fitter is iterative, show iteration number.

        formula: str
            an R-like formula


        Returns
        -------
        self: AalenAdditiveFitter
            self with additional new properties: ``cumulative_hazards_``, etc.

        Examples
        --------
        .. code:: python

            from lifelines import AalenAdditiveFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            aaf = AalenAdditiveFitter()
            aaf.fit(df, 'T', 'E')
            aaf.predict_median(df)
            aaf.print_summary()

        """
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        df = df.copy()

        self.duration_col = duration_col
        self.event_col = event_col
        self.weights_col = weights_col
        self.formula = formula

        self._n_examples = df.shape[0]

        X, T, E, weights = self._preprocess_dataframe(df)

        self.durations = T.copy()
        self.event_observed = E.copy()
        self.weights = weights.copy()

        self._norm_std = X.std(0)

        # if we included an intercept, we need to fix not divide by zero.
        if self.fit_intercept:
            self._norm_std["Intercept"] = 1.0
        else:
            # a _intercept was provided
            self._norm_std[self._norm_std < 1e-8] = 1.0

        self.hazards_, self.cumulative_hazards_, self.cumulative_variance_ = self._fit_model(
            normalize(X, 0, self._norm_std), T, E, weights, show_progress
        )
        self.hazards_ /= self._norm_std
        self.cumulative_hazards_ /= self._norm_std
        self.cumulative_variance_ /= self._norm_std
        self.confidence_intervals_ = self._compute_confidence_intervals()

        self._index = self.hazards_.index

        self._predicted_hazards_ = self.predict_cumulative_hazard(df).iloc[-1].values.ravel()
        return self

    def _fit_model(self, X, T, E, weights, show_progress):

        columns = X.columns
        index = np.sort(np.unique(T[E]))

        hazards_, variance_hazards_, stop = self._fit_model_to_data_batch(
            X.values, T.values, E.values, weights.values, show_progress
        )

        hazards = pd.DataFrame(hazards_, columns=columns, index=index).iloc[:stop]
        cumulative_hazards_ = hazards.cumsum()
        cumulative_variance_hazards_ = pd.DataFrame(variance_hazards_, columns=columns, index=index).iloc[:stop].cumsum()

        return hazards, cumulative_hazards_, cumulative_variance_hazards_

    def _fit_model_to_data_batch(self, X, T, E, weights, show_progress):

        n, d = X.shape

        # we are mutating values of X, so copy it.
        X = X.copy()

        # iterate over all the unique death times
        unique_death_times = np.sort(np.unique(T[E]))
        n_deaths = unique_death_times.shape[0]
        total_observed_exits = 0

        hazards_ = np.zeros((n_deaths, d))
        variance_hazards_ = np.zeros((n_deaths, d))
        v = np.zeros(d)
        start = time.time()
        last_iteration = 0

        W = np.sqrt(weights)
        X = W[:, None] * X

        for i, t in enumerate(unique_death_times):

            exits = T == t
            deaths = exits & E
            try:
                v, V = lr(X, W * deaths, c1=self.coef_penalizer, c2=self.smoothing_penalizer, offset=v, ix=deaths)
            except LinAlgError:
                warnings.warn(
                    "Linear regression error at index=%d, time=%.3f. Try increasing the coef_penalizer value." % (i, t),
                    ConvergenceWarning,
                )
                v = np.zeros_like(v)
                V = np.zeros_like(V)

            hazards_[i, :] = v

            variance_hazards_[i, :] = (V ** 2).sum(1)

            X[exits, :] = 0

            if show_progress and i % int((n_deaths / 10)) == 0:
                print("\rIteration %d/%d, seconds_since_start = %.2f" % (i + 1, n_deaths, time.time() - start))

            last_iteration = i + 1
            # terminate early when there are less than (3 * d) subjects left, where d does not include the intercept.
            # the value 3 if from R survival lib.
            if (3 * (d - 1)) >= n - total_observed_exits:
                if show_progress:
                    print("Terminating early due to too few subjects remaining. This is expected behaviour.")
                break

            total_observed_exits += exits.sum()

        if show_progress:
            print("Convergence completed.")
        return hazards_, variance_hazards_, last_iteration

    def _preprocess_dataframe(self, df):
        n, _ = df.shape

        df = df.sort_values(by=self.duration_col)

        # Extract time and event
        T = df.pop(self.duration_col)
        E = df.pop(self.event_col) if (self.event_col is not None) else pd.Series(np.ones(n), index=df.index, name="E")
        W = (
            df.pop(self.weights_col)
            if (self.weights_col is not None)
            else pd.Series(np.ones((n,)), index=df.index, name="weights")
        )

        # check to make sure their weights are okay
        if self.weights_col:
            if (W.astype(int) != W).any():
                warnings.warn(
                    """It appears your weights are not integers, possibly propensity or sampling scores then?
It's important to know that the naive variance estimates of the coefficients are biased."
""",
                    StatisticalWarning,
                )
            if (W <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

        self.regressors = utils.CovariateParameterMappings({"beta_": self.formula}, df, force_intercept=self.fit_intercept)
        X = self.regressors.transform_df(df)["beta_"]

        T = T.astype(float)

        check_nans_or_infs(E)
        E = E.astype(bool)

        self._check_values(X, T, E)

        return X, T, E, W

    def predict_cumulative_hazard(self, X):
        """
        Returns the hazard rates for the individuals

        Parameters
        ----------
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        """

        cols = _get_index(X)
        if isinstance(X, pd.DataFrame):
            X = self.regressors.transform_df(X)["beta_"]
        elif isinstance(X, pd.Series):
            return self.predict_cumulative_hazard(X.to_frame().T.infer_objects())

        X = X.astype(float)

        timeline = self._index
        individual_cumulative_hazards_ = pd.DataFrame(np.dot(self.cumulative_hazards_, X.T), index=timeline, columns=cols)

        return individual_cumulative_hazards_

    def _check_values(self, X, T, E):
        check_for_numeric_dtypes_or_raise(X)
        check_nans_or_infs(T)
        check_nans_or_infs(X)

    def predict_survival_function(self, X, times=None):
        """
        Returns the survival functions for the individuals

        Parameters
        ----------
        X: a (n,d) covariate numpy array or DataFrame
            If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        times:
            Not implemented yet

        """
        return np.exp(-self.predict_cumulative_hazard(X))

    def predict_percentile(self, X, p=0.5) -> pd.Series:
        """
        Returns the median lifetimes for the individuals.
        http://stats.stackexchange.com/questions/102986/percentile-loss-functions

        Parameters
        ----------
        X: a (n,d) covariate numpy array or DataFrame
            If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        p: float
            default: 0.5

        """
        index = _get_index(X)
        return qth_survival_times(p, self.predict_survival_function(X)[index]).T.squeeze()

    def predict_median(self, X) -> pd.Series:
        """

        Parameters
        ----------
        X: a (n,d) covariate numpy array or DataFrame
            If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the median lifetimes for the individuals
        """
        return self.predict_percentile(X, 0.5)

    def predict_expectation(self, X) -> pd.Series:
        """
        Compute the expected lifetime, E[T], using covariates X.

        Parameters
        ----------
        X: a (n,d) covariate numpy array or DataFrame
            If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns the expected lifetimes for the individuals
        """
        index = _get_index(X)
        t = self._index
        return pd.Series(trapezoid(self.predict_survival_function(X)[index].values.T, t), index=index)

    def _compute_confidence_intervals(self):
        ci = 100 * (1 - self.alpha)
        z = inv_normal_cdf(1 - self.alpha / 2)
        std_error = np.sqrt(self.cumulative_variance_)
        return pd.concat(
            {
                "%g%% lower-bound" % ci: self.cumulative_hazards_ - z * std_error,
                "%g%% upper-bound" % ci: self.cumulative_hazards_ + z * std_error,
            }
        )

    def plot(self, columns=None, loc=None, iloc=None, ax=None, **kwargs):
        """ "
        A wrapper around plotting. Matplotlib plot arguments can be passed in, plus:

        Parameters
        -----------
        columns: string or list-like, optional
          If not empty, plot a subset of columns from the ``cumulative_hazards_``. Default all.
        loc:

        iloc: slice, optional
          specify a location-based subsection of the curves to plot, ex:
                 ``.plot(iloc=slice(0,10))`` will plot the first 10 time points.
        """
        from matplotlib import pyplot as plt

        assert loc is None or iloc is None, "Cannot set both loc and iloc in call to .plot"

        def shaded_plot(ax, x, y, y_upper, y_lower, **kwargs):
            (base_line,) = ax.plot(x, y, drawstyle="steps-post", **kwargs)
            ax.fill_between(x, y_lower, y2=y_upper, alpha=0.25, color=base_line.get_color(), linewidth=1.0, step="post")

        def create_df_slicer(loc, iloc):
            get_method = "loc" if loc is not None else "iloc"

            if iloc is None and loc is None:
                user_submitted_ix = slice(0, None)
            else:
                user_submitted_ix = loc if loc is not None else iloc

            return lambda df: getattr(df, get_method)[user_submitted_ix]

        subset_df = create_df_slicer(loc, iloc)

        if not columns:
            columns = self.cumulative_hazards_.columns
        else:
            columns = _to_list(columns)

        if ax is None:
            ax = plt.gca()

        x = subset_df(self.cumulative_hazards_).index.values.astype(float)

        for column in columns:
            ci = (1 - self.alpha) * 100
            y = subset_df(self.cumulative_hazards_[column]).values
            index = subset_df(self.cumulative_hazards_[column]).index
            y_upper = subset_df(self.confidence_intervals_[column].loc["%g%% upper-bound" % ci]).values
            y_lower = subset_df(self.confidence_intervals_[column].loc["%g%% lower-bound" % ci]).values
            shaded_plot(ax, x, y, y_upper, y_lower, label=column, **kwargs)

        plt.hlines(0, index.min() - 1, index.max(), color="k", linestyles="--", alpha=0.5)

        ax.legend()
        return ax

    def smoothed_hazards_(self, bandwidth=1):
        """
        Using the epanechnikov kernel to smooth the hazard function, with sigma/bandwidth

        """
        timeline = self._index.values
        return pd.DataFrame(
            np.dot(epanechnikov_kernel(timeline[:, None], timeline, bandwidth), self.hazards_.values),
            columns=self.hazards_.columns,
            index=timeline,
        )

    @property
    def concordance_index_(self):
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the ROC AUC
        to survival data, including censorships.

        For this purpose, the ``score_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset. It's analogous to the R^2 in linear models.

        """
        # pylint: disable=access-member-before-definition
        if hasattr(self, "_predicted_hazards_"):
            self._concordance_score_ = concordance_index(self.durations, -self._predicted_hazards_, self.event_observed)
            del self._predicted_hazards_
            return self._concordance_score_
        return self._concordance_score_

    def _compute_slopes(self):
        def _univariate_linear_regression_without_intercept(X, Y, weights):
            # normally (weights * X).dot(Y) / X.dot(weights * X), but we have a slightly different form here.
            beta = X.dot(Y) / X.dot(weights * X)
            errors = Y.values - np.outer(X, beta)
            var = (errors ** 2).sum(0) / (Y.shape[0] - 2) / X.dot(weights * X)
            return beta, np.sqrt(var)

        weights = survival_table_from_events(self.durations, self.event_observed).loc[self._index, "at_risk"].values
        y = (weights[:, None] * self.hazards_).cumsum()
        X = self._index.values
        betas, se = _univariate_linear_regression_without_intercept(X, y, weights)
        return pd.Series(betas, index=y.columns), pd.Series(se, index=y.columns)

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
        """
        df = pd.DataFrame(index=self.cumulative_hazards_.columns)

        betas, se = self._compute_slopes()
        df["slope(coef)"] = betas
        df["se(slope(coef))"] = se
        return df

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
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """
        justify = string_rjustify(25)

        headers = []
        headers.append(("duration col", "'%s'" % self.duration_col))

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if self.coef_penalizer > 0:
            headers.append(("coef penalizer", self.coef_penalizer))
        if self.smoothing_penalizer > 0:
            headers.append(("smoothing penalizer", self.smoothing_penalizer))

        headers.extend(
            [
                ("number of subjects", self._n_examples),
                ("number of events observed", self.event_observed.sum()),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        footers = [("Concordance", "{:.{prec}f}".format(self.concordance_index_, prec=decimals))]
        p = Printer(self, headers, footers, justify, kwargs, decimals, columns)

        p.print(style=style)

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
        """
        if scoring_method == "log_likelihood":
            raise NotImplementedError("Only concordance_index is available")

        T = df.pop(self.duration_col).astype(float)
        E = df.pop(self.event_col).astype(bool)

        predictions = self.predict_median(df)
        return concordance_index(T, predictions, event_observed=E)
