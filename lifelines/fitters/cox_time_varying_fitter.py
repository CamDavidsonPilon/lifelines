# -*- coding: utf-8 -*-


from datetime import datetime
import warnings
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from numpy.linalg import norm, inv
from numpy import sum as array_sum_to_scalar

from scipy.linalg import solve as spsolve, LinAlgError

from autograd import elementwise_grad
from autograd import numpy as anp


from lifelines.fitters import SemiParametricRegressionFitter
from lifelines.fitters.mixins import ProportionalHazardMixin
from lifelines.utils.printer import Printer
from lifelines.statistics import _chisq_test_p_value, StatisticalResult
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.utils import (
    _get_index,
    _to_list,
    # check_for_overlapping_intervals,
    check_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation_low_variance,
    check_for_immediate_deaths,
    check_for_instantaneous_events_at_time_zero,
    check_for_instantaneous_events_at_death_time,
    check_for_nonnegative_intervals,
    pass_for_numeric_dtypes_or_raise_array,
    inv_normal_cdf,
    normalize,
    StepSizer,
    check_nans_or_infs,
    string_rjustify,
    coalesce,
)
from lifelines import utils

__all__ = ["CoxTimeVaryingFitter"]

matrix_axis_0_sum_to_1d_array = lambda m: np.sum(m, 0)


class CoxTimeVaryingFitter(SemiParametricRegressionFitter, ProportionalHazardMixin):
    r"""
    This class implements fitting Cox's time-varying proportional hazard model:

        .. math::  h(t|x(t)) = h_0(t)\exp((x(t)-\overline{x})'\beta)

    Parameters
    ----------
    alpha: float, optional (default=0.05)
       the level in the confidence intervals.
    penalizer: float, optional
        the coefficient of an L2 penalizer in the regression

    Attributes
    ----------
    params_ : Series
        The estimated coefficients. Changed in version 0.22.0: use to be ``.hazards_``
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    strata: list | str
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """

    _KNOWN_MODEL = True

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio: float = 0.0, strata=None):
        super(CoxTimeVaryingFitter, self).__init__(alpha=alpha)
        self.alpha = alpha
        self.penalizer = penalizer
        self.strata = utils._to_list_or_singleton(strata)
        self.l1_ratio = l1_ratio

    def fit(
        self,
        df,
        event_col,
        start_col="start",
        stop_col="stop",
        weights_col=None,
        id_col=None,
        show_progress=False,
        robust=False,
        strata=None,
        initial_point=None,
        formula: str = None,
        fit_options: Optional[dict] = None,
    ):  # pylint: disable=too-many-arguments
        """
        Fit the Cox Proportional Hazard model to a time varying dataset. Tied survival times
        are handled using Efron's tie-method.

        Parameters
        -----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
           `event_col`, plus other covariates. `duration_col` refers to
           the lifetimes of the subjects. `event_col` refers to whether
           the 'death' events was observed: 1 if observed, 0 else (censored).
        event_col: string
           the column in DataFrame that contains the subjects' death
           observation. If left as None, assume all individuals are non-censored.
        start_col: string
            the column that contains the start of a subject's time period.
        stop_col: string
            the column that contains the end of a subject's time period.
        weights_col: string, optional
            the column that contains (possibly time-varying) weight of each subject-period row.
        id_col: string, optional
            A subject could have multiple rows in the DataFrame. This column contains
           the unique identifier per subject. If not provided, it's up to the
           user to make sure that there are no violations.
        show_progress: since the fitter is iterative, show convergence
           diagnostics.
        robust: bool, optional (default: True)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
          ties, so if there are high number of ties, results may significantly differ. See
          "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078
        strata: list | string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        formula: str, optional
            A R-like formula for transforming the covariates
        fit_options: dict, optional
            Override the default values in NR algorithm:
                step_size: 0.95,
                precision: 1e-07,
                r_precision=1e-9,
                max_steps: 500,

        Returns
        --------
        self: CoxTimeVaryingFitter
            self, with additional properties like ``hazards_`` and ``print_summary``

        """
        self.strata = utils._to_list_or_singleton(coalesce(strata, self.strata))
        self.robust = robust
        if self.robust:
            raise NotImplementedError("Not available yet.")

        self.event_col = event_col
        self.id_col = id_col
        self.stop_col = stop_col
        self.start_col = start_col
        self.formula = formula
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        df = df.copy()

        if not (event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the DataFrame provided.")

        if weights_col is None:
            self.weights_col = None
            assert "__weights" not in df.columns, "__weights is an internal lifelines column, please rename your column first."
            df["__weights"] = 1.0
        else:
            self.weights_col = weights_col
            if (df[weights_col] <= 0).any():
                raise ValueError("values in weights_col must be positive.")

        df = df.rename(columns={event_col: "event", start_col: "start", stop_col: "stop", weights_col: "__weights"})

        if self.strata is not None and self.id_col is not None:
            df = df.set_index(_to_list(self.strata) + [id_col])
            df = df.sort_index()
        elif self.strata is not None and self.id_col is None:
            df = df.set_index(_to_list(self.strata))
        elif self.strata is None and self.id_col is not None:
            df = df.set_index([id_col])

        events, start, stop = (
            pass_for_numeric_dtypes_or_raise_array(df.pop("event")).astype(bool),
            df.pop("start"),
            df.pop("stop"),
        )
        weights = df.pop("__weights").astype(float)

        self.regressors = utils.CovariateParameterMappings({"beta_": self.formula}, df, force_no_intercept=True)
        X = self.regressors.transform_df(df)["beta_"]

        self._check_values(X, events, start, stop)

        self._norm_mean = X.mean(0)
        self._norm_std = X.std(0)

        beta_, self.log_likelihood_, self._hessian_ = self._newton_raphson_for_efron_model(
            normalize(X, self._norm_mean, self._norm_std),
            events,
            start,
            stop,
            weights,
            initial_point=initial_point,
            show_progress=show_progress,
            **utils.coalesce(fit_options, {}),
        )

        self.params_ = pd.Series(beta_, index=pd.Index(X.columns, name="covariate"), name="coef") / self._norm_std
        if self._hessian_.size > 0:
            # possible if the df is trivial (no covariate columns)
            self.variance_matrix_ = pd.DataFrame(-inv(self._hessian_) / np.outer(self._norm_std, self._norm_std), index=X.columns)

        else:
            self.variance_matrix_ = pd.DataFrame(index=X.columns, columns=X.columns)

        self.standard_errors_ = self._compute_standard_errors(
            normalize(X, self._norm_mean, self._norm_std), events, start, stop, weights
        )
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_cumulative_hazard_ = self._compute_cumulative_baseline_hazard(df, events, start, stop, weights)
        self.baseline_survival_ = self._compute_baseline_survival()
        self.event_observed = events
        self.start_stop_and_events = pd.DataFrame({"event": events, "start": start, "stop": stop})
        self.weights = weights

        self._n_examples = X.shape[0]
        self._n_unique = X.index.unique().shape[0]
        return self

    def _check_values(self, df, events, start, stop):
        # check_for_overlapping_intervals(df) # this is currently too slow for production.
        check_nans_or_infs(df)
        check_low_var(df)
        check_complete_separation_low_variance(df, events, self.event_col)
        check_for_numeric_dtypes_or_raise(df)
        check_for_nonnegative_intervals(start, stop)
        check_for_immediate_deaths(events, start, stop)
        check_for_instantaneous_events_at_time_zero(start, stop)
        check_for_instantaneous_events_at_death_time(events, start, stop)

    def _partition_by_strata(self, X, events, start, stop, weights):
        for stratum, stratified_X in X.groupby(self.strata):
            stratified_W = weights.loc[stratum]
            stratified_start = start.loc[stratum]
            stratified_events = events.loc[stratum]
            stratified_stop = stop.loc[stratum]
            yield (
                stratified_X.values,
                stratified_events.values,
                stratified_start.values,
                stratified_stop.values,
                stratified_W.values,
            ), stratum

    def _partition_by_strata_and_apply(self, X, events, start, stop, weights, function, *args):
        for ((stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W), _) in self._partition_by_strata(
            X, events, start, stop, weights
        ):
            yield function(stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W, *args)

    def _compute_z_values(self):
        return self.params_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_confidence_intervals(self):
        ci = 100 * (1 - self.alpha)
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        hazards = self.params_.values
        return pd.DataFrame(
            np.c_[hazards - z * se, hazards + z * se],
            columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
            index=self.params_.index,
        )

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
        ci = 100 * (1 - self.alpha)
        z = inv_normal_cdf(1 - self.alpha / 2)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = self.hazard_ratios_
            df["se(coef)"] = self.standard_errors_
            df["coef lower %g%%" % ci] = self.confidence_intervals_["%g%% lower-bound" % ci]
            df["coef upper %g%%" % ci] = self.confidence_intervals_["%g%% upper-bound" % ci]
            df["exp(coef) lower %g%%" % ci] = self.hazard_ratios_ * np.exp(-z * self.standard_errors_)
            df["exp(coef) upper %g%%" % ci] = self.hazard_ratios_ * np.exp(z * self.standard_errors_)
            df["cmp to"] = np.zeros_like(self.params_)
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -utils.quiet_log2(df["p"])
            return df

    def _newton_raphson_for_efron_model(
        self,
        df,
        events,
        start,
        stop,
        weights,
        show_progress=False,
        step_size=0.95,
        precision=1e-8,
        r_precision=1e-9,
        max_steps=50,
        initial_point=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        """
        Newton Raphson algorithm for fitting CPH model.

        Parameters
        ----------
        df: DataFrame
        stop_times_events: DataFrame
             meta information about the subjects history
        show_progress: bool, optional (default: True)
            to show verbose output of convergence
        step_size: float
            > 0 to determine a starting step size in NR algorithm.
        precision: float
            the algorithm stops if the norm of delta between
            successive positions is less than ``precision``.
        r_precision: float, optional
            the algorithms stops if the relative decrease in log-likelihood
            between successive iterations goes below ``r_precision``.

        Returns
        --------
        beta: (1,d) numpy array.
        """
        assert precision <= 1.0, "precision must be less than or equal to 1."

        # soft penalizer functions, from https://www.cs.ubc.ca/cgi-bin/tr/2009/TR-2009-19.pdf
        soft_abs = lambda x, a: 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))
        penalizer = (
            lambda beta, a: n
            * (self.penalizer * (self.l1_ratio * (soft_abs(beta, a)) + 0.5 * (1 - self.l1_ratio) * (beta**2))).sum()
        )
        d_penalizer = elementwise_grad(penalizer)
        dd_penalizer = elementwise_grad(d_penalizer)

        n, d = df.shape

        # make sure betas are correct size.
        if initial_point is not None:
            beta = initial_point
        else:
            beta = np.zeros((d,))

        i = 0
        converging = True
        ll, previous_ll = 0, 0
        start_time = time.time()

        step_sizer = StepSizer(step_size)
        step_size = step_sizer.next()

        while converging:
            i += 1

            if self.strata is None:
                h, g, ll = self._get_gradients(df.values, events.values, start.values, stop.values, weights.values, beta)
            else:
                g = np.zeros_like(beta)
                h = np.zeros((d, d))
                ll = 0
                for _h, _g, _ll in self._partition_by_strata_and_apply(
                    df, events, start, stop, weights, self._get_gradients, beta
                ):
                    g += _g
                    h += _h
                    ll += _ll

            if i == 1 and np.all(beta == 0):
                # this is a neat optimization, the null partial likelihood
                # is the same as the full partial but evaluated at zero.
                # if the user supplied a non-trivial initial point, we need to delay this.
                self._log_likelihood_null = ll

            if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
                ll -= penalizer(beta, 1.5**i)
                g -= d_penalizer(beta, 1.5**i)
                h[np.diag_indices(d)] -= dd_penalizer(beta, 1.5**i)

            try:
                # reusing a piece to make g * inv(h) * g.T faster later
                inv_h_dot_g_T = spsolve(-h, g, assume_a="pos")
            except ValueError as e:
                if "infs or NaNs" in str(e):
                    raise ConvergenceError(
                        """hessian or gradient contains nan or inf value(s). Convergence halted. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
""",
                        e,
                    )
                else:
                    # something else?
                    raise e
            except LinAlgError as e:
                raise ConvergenceError(
                    """Convergence halted due to matrix inversion problems. Suspicion is high colinearity. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
""",
                    e,
                )

            delta = step_size * inv_h_dot_g_T

            if np.any(np.isnan(delta)):
                raise ConvergenceError(
                    """delta contains nan value(s). Convergence halted. Please see the following tips in the lifelines documentation:
https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model
"""
                )
            # Save these as pending result
            hessian, gradient = h, g
            norm_delta = norm(delta)
            newton_decrement = g.dot(inv_h_dot_g_T) / 2

            if show_progress:
                print(
                    "\rIteration %d: norm_delta = %.2e, step_size = %.4f, log_lik = %.5f, newton_decrement = %.2e, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll, newton_decrement, time.time() - start_time)
                )

            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif previous_ll > 0 and abs(ll - previous_ll) / (-previous_ll) < r_precision:
                # this is what R uses by default with r_precision=1e-9
                converging, completed = False, True
            elif newton_decrement < precision:
                converging, completed = False, True
            elif i >= max_steps:
                # 50 iterations steps with N-R is a lot.
                # Expected convergence is less than 10 steps
                converging, completed = False, False
            elif step_size <= 0.0001:
                converging, completed = False, False
            elif abs(ll) < 0.0001 and norm_delta > 1.0:
                warnings.warn(
                    "The log-likelihood is getting suspiciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression\n",
                    ConvergenceWarning,
                )
                converging, completed = False, False

            step_size = step_sizer.update(norm_delta).next()

            beta += delta

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        elif show_progress and not completed:
            print("Convergence failed. See any warning messages.")

        # report to the user problems that we detect.
        if completed and norm_delta > 0.1:
            warnings.warn(
                "Newton-Raphson convergence completed but norm(delta) is still high, %.3f. This may imply non-unique solutions to the maximum likelihood. Perhaps there is colinearity or complete separation in the dataset?"
                % norm_delta,
                ConvergenceWarning,
            )
        elif not completed:
            warnings.warn("Newton-Raphson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta, ll, hessian

    @staticmethod
    def _get_gradients(X, events, start, stop, weights, beta):  # pylint: disable=too-many-locals
        """
        Calculates the first and second order vector differentials, with respect to beta.

        Returns
        -------
        hessian: (d, d) numpy array,
        gradient: (1, d) numpy array
        log_likelihood: float
        """

        _, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros(d)
        log_lik = 0
        unique_death_times = np.unique(stop[events])

        for t in unique_death_times:

            # I feel like this can be made into some tree-like structure
            ix = (start < t) & (t <= stop)

            X_at_t = X[ix]
            weights_at_t = weights[ix]
            stops_events_at_t = stop[ix]
            events_at_t = events[ix]

            phi_i = weights_at_t * np.exp(np.dot(X_at_t, beta))
            phi_x_i = phi_i[:, None] * X_at_t
            phi_x_x_i = np.dot(X_at_t.T, phi_x_i)

            # Calculate sums of Risk set
            risk_phi = array_sum_to_scalar(phi_i)
            risk_phi_x = matrix_axis_0_sum_to_1d_array(phi_x_i)
            risk_phi_x_x = phi_x_x_i

            # Calculate the sums of Tie set
            deaths = events_at_t & (stops_events_at_t == t)

            tied_death_counts = array_sum_to_scalar(deaths.astype(int))  # should always at least 1. Why? TODO

            xi_deaths = X_at_t[deaths]

            x_death_sum = matrix_axis_0_sum_to_1d_array(weights_at_t[deaths, None] * xi_deaths)

            weight_count = array_sum_to_scalar(weights_at_t[deaths])
            weighted_average = weight_count / tied_death_counts

            #
            # This code is near identical to the _batch algorithm in CoxPHFitter. In fact, see _batch for comments.
            #

            if tied_death_counts > 1:

                # A good explanation for how Efron handles ties. Consider three of five subjects who fail at the time.
                # As it is not known a priori that who is the first to fail, so one-third of
                # (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
                # of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

                # a lot of this is now in Einstein notation for performance, but see original "expanded" code here
                # https://github.com/CamDavidsonPilon/lifelines/blob/e7056e7817272eb5dff5983556954f56c33301b1/lifelines/fitters/cox_time_varying_fitter.py#L458-L490

                tie_phi = array_sum_to_scalar(phi_i[deaths])
                tie_phi_x = matrix_axis_0_sum_to_1d_array(phi_x_i[deaths])
                tie_phi_x_x = np.dot(xi_deaths.T, phi_i[deaths, None] * xi_deaths)

                increasing_proportion = np.arange(tied_death_counts) / tied_death_counts
                denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)
                numer = risk_phi_x - np.outer(increasing_proportion, tie_phi_x)

                a1 = np.einsum("ab, i->ab", risk_phi_x_x, denom) - np.einsum(
                    "ab, i->ab", tie_phi_x_x, increasing_proportion * denom
                )
            else:
                # no tensors here, but do some casting to make it easier in the converging step next.
                denom = 1.0 / np.array([risk_phi])
                numer = risk_phi_x
                a1 = risk_phi_x_x * denom

            summand = numer * denom[:, None]
            a2 = summand.T.dot(summand)

            gradient = gradient + x_death_sum - weighted_average * summand.sum(0)
            log_lik = log_lik + np.dot(x_death_sum, beta) + weighted_average * np.log(denom).sum()
            hessian = hessian + weighted_average * (a2 - a1)

        return hessian, gradient, log_lik

    def predict_log_partial_hazard(self, X) -> pd.Series:
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - \bar{x})'\beta`


        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame

        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        hazard_names = self.params_.index

        if isinstance(X, pd.DataFrame):
            X = self.regressors.transform_df(X)["beta_"]
            X = X.values

        X = X.astype(float)
        index = _get_index(X)
        X = normalize(X, self._norm_mean.values, 1)
        return pd.Series(np.dot(X, self.params_), index=index)

    def predict_partial_hazard(self, X) -> pd.Series:
        r"""
        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`\exp{(x - \bar{x})'\beta }`

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame

        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        """
        return np.exp(self.predict_log_partial_hazard(X))

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
        justify = string_rjustify(18)

        headers = []

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
            headers.append(("penalizer", self.penalizer))
        if self.strata:
            headers.append(("strata", self.strata))

        headers.extend(
            [
                ("number of subjects", self._n_unique),
                ("number of periods", self._n_examples),
                ("number of events", self.event_observed.sum()),
                ("partial log-likelihood", "{:.{prec}f}".format(self.log_likelihood_, prec=decimals)),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        sr = self.log_likelihood_ratio_test()
        footers = []
        footers.extend(
            [
                ("Partial AIC", "{:.{prec}f}".format(self.AIC_partial_, prec=decimals)),
                (
                    "log-likelihood ratio test",
                    "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                ),
                ("-log2(p) of ll-ratio test", "{:.{prec}f}".format(-utils.quiet_log2(sr.p_value), prec=decimals)),
            ]
        )

        p = Printer(self, headers, footers, justify, kwargs, decimals, columns)
        p.print(style=style)

    def log_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Cox model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.

        Conveniently, we can actually use CoxPHFitter class to do most of the work.

        """
        if hasattr(self, "_log_likelihood_null"):
            ll_null = self._log_likelihood_null

        else:
            trivial_dataset = self.start_stop_and_events
            trivial_dataset = trivial_dataset.join(self.weights)
            trivial_dataset = trivial_dataset.reset_index()
            ll_null = (
                CoxTimeVaryingFitter()
                .fit(
                    trivial_dataset,
                    start_col=self.start_col,
                    stop_col=self.stop_col,
                    event_col=self.event_col,
                    id_col=self.id_col,
                    weights_col="__weights",
                    strata=self.strata,
                )
                .log_likelihood_
            )

        ll_alt = self.log_likelihood_
        test_stat = 2 * (ll_alt - ll_null)
        degrees_freedom = self.params_.shape[0]
        p_value = _chisq_test_p_value(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value, test_stat, name="log-likelihood ratio test", degrees_freedom=degrees_freedom, null_distribution="chi squared"
        )

    def plot(self, columns=None, ax=None, **errorbar_kwargs):
        """
        Produces a visual representation of the coefficients, including their standard errors and magnitudes.

        Parameters
        ----------
        columns : list, optional
            specify a subset of the columns to plot
        errorbar_kwargs:
            pass in additional plotting commands to matplotlib errorbar command

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

        z = inv_normal_cdf(1 - self.alpha / 2)

        if columns is None:
            user_supplied_columns = False
            columns = self.params_.index
        else:
            user_supplied_columns = True

        yaxis_locations = list(range(len(columns)))
        symmetric_errors = z * self.standard_errors_[columns].values.copy()
        hazards = self.params_[columns].values.copy()

        order = list(range(len(columns) - 1, -1, -1)) if user_supplied_columns else np.argsort(hazards)

        ax.errorbar(hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
        ax.set_ylim(best_ylim)

        tick_labels = [columns[i] for i in order]

        ax.set_yticks(yaxis_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def _compute_cumulative_baseline_hazard(self, tv_data, events, start, stop, weights):  # pylint: disable=too-many-locals

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hazards = self.predict_partial_hazard(tv_data).values

        unique_death_times = np.unique(stop[events.values])
        baseline_hazard_ = pd.DataFrame(np.zeros_like(unique_death_times).astype(float), index=unique_death_times, columns=["baseline hazard"])

        for t in unique_death_times:
            ix = (start.values < t) & (t <= stop.values)

            events_at_t = events.values[ix]
            stops_at_t = stop.values[ix]
            weights_at_t = weights.values[ix]
            hazards_at_t = hazards[ix]

            deaths = events_at_t & (stops_at_t == t)

            death_counts = (weights_at_t.squeeze() * deaths).sum()  # should always be atleast 1.
            baseline_hazard_.loc[t] = death_counts / hazards_at_t.sum()

        return baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        survival_df = np.exp(-self.baseline_cumulative_hazard_)
        survival_df.columns = ["baseline survival"]
        return survival_df

    def __repr__(self):
        classname = self._class_name
        try:
            s = """<lifelines.%s: fitted with %d periods, %d subjects, %d events>""" % (
                classname,
                self._n_examples,
                self._n_unique,
                self.event_observed.sum(),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

    def _compute_residuals(self, df, events, start, stop, weights):
        raise NotImplementedError()

    def _compute_delta_beta(self, df, events, start, stop, weights):
        """approximate change in betas as a result of excluding ith row"""

        score_residuals = self._compute_residuals(df, events, start, stop, weights) * weights[:, None]

        naive_var = inv(self._hessian_)
        delta_betas = -score_residuals.dot(naive_var) / self._norm_std.values

        return delta_betas

    def _compute_sandwich_estimator(self, X, events, start, stop, weights):

        delta_betas = self._compute_delta_beta(X, events, start, stop, weights)

        if self.cluster_col:
            delta_betas = pd.DataFrame(delta_betas).groupby(self._clusters).sum().values

        sandwich_estimator = delta_betas.T.dot(delta_betas)
        return sandwich_estimator

    def _compute_standard_errors(self, X, events, start, stop, weights):
        if self.robust:
            se = np.sqrt(self._compute_sandwich_estimator(X, events, start, stop, weights).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.values.diagonal())
        return pd.Series(se, index=self.params_.index, name="se")
