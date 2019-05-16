# -*- coding: utf-8 -*-


from datetime import datetime
import warnings
import time

import numpy as np
import pandas as pd
from scipy import stats

from numpy.linalg import norm, inv
from scipy.linalg import solve as spsolve, LinAlgError


from bottleneck import nansum as array_sum_to_scalar

from lifelines.fitters import BaseFitter
from lifelines.statistics import chisq_test, StatisticalResult
from lifelines.utils import (
    _get_index,
    _to_list,
    # check_for_overlapping_intervals,
    check_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation_low_variance,
    check_for_immediate_deaths,
    check_for_instantaneous_events,
    pass_for_numeric_dtypes_or_raise_array,
    ConvergenceError,
    ConvergenceWarning,
    inv_normal_cdf,
    normalize,
    StepSizer,
    check_nans_or_infs,
    string_justify,
    format_p_value,
    format_exp_floats,
    format_floats,
    coalesce,
)

__all__ = ["CoxTimeVaryingFitter"]

matrix_axis_0_sum_to_array = lambda m: np.sum(m, 0)


class CoxTimeVaryingFitter(BaseFitter):
    r"""
    This class implements fitting Cox's time-varying proportional hazard model:

        .. math::  h(t|x(t)) = h_0(t)\exp(x(t)'\beta)

    Parameters
    ----------
    alpha: float, optional (default=0.05)
       the level in the confidence intervals.
    penalizer: float, optional
        the coefficient of an L2 penalizer in the regression

    Attributes
    ----------
    hazards_ : Series
        The estimated hazards
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    score_: float
        the concordance index of the model.
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """

    def __init__(self, alpha=0.05, penalizer=0.0, strata=None):
        super(CoxTimeVaryingFitter, self).__init__(alpha=alpha)
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")

        self.alpha = alpha
        self.penalizer = penalizer
        self.strata = strata

    def fit(
        self,
        df,
        id_col,
        event_col,
        start_col="start",
        stop_col="stop",
        weights_col=None,
        show_progress=False,
        step_size=None,
        robust=False,
        strata=None,
        initial_point=None,
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
        id_col: string
            A subject could have multiple rows in the DataFrame. This column contains
           the unique identifier per subject.
        event_col: string
           the column in DataFrame that contains the subjects' death
           observation. If left as None, assume all individuals are non-censored.
        start_col: string
            the column that contains the start of a subject's time period.
        stop_col: string
            the column that contains the end of a subject's time period.
        weights_col: string, optional
            the column that contains (possibly time-varying) weight of each subject-period row.
        show_progress: since the fitter is iterative, show convergence
           diagnostics.
        robust: boolean, optional (default: True)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
          ties, so if there are high number of ties, results may significantly differ. See
          "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078
        step_size: float, optional
            set an initial step size for the fitting algorithm.
        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        Returns
        --------
        self: CoxTimeVaryingFitter
            self, with additional properties like ``hazards_`` and ``print_summary``

        """
        self.strata = coalesce(strata, self.strata)
        self.robust = robust
        if self.robust:
            raise NotImplementedError("Not available yet.")

        self.event_col = event_col
        self.id_col = id_col
        self.stop_col = stop_col
        self.start_col = start_col
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        df = df.copy()

        if not (id_col in df and event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the DataFrame provided.")

        if weights_col is None:
            self.weights_col = None
            assert (
                "__weights" not in df.columns
            ), "__weights is an internal lifelines column, please rename your column first."
            df["__weights"] = 1.0
        else:
            self.weights_col = weights_col
            if (df[weights_col] <= 0).any():
                raise ValueError("values in weights_col must be positive.")

        df = df.rename(
            columns={id_col: "id", event_col: "event", start_col: "start", stop_col: "stop", weights_col: "__weights"}
        )

        if self.strata is None:
            df = df.set_index("id")
        else:
            df = df.set_index(_to_list(self.strata) + ["id"])  # TODO: needs to be a list
            df = df.sort_index()

        events, start, stop = (
            pass_for_numeric_dtypes_or_raise_array(df.pop("event")).astype(bool),
            df.pop("start"),
            df.pop("stop"),
        )
        weights = df.pop("__weights").astype(float)

        df = df.astype(float)
        self._check_values(df, events, start, stop)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)

        hazards_ = self._newton_rhaphson(
            normalize(df, self._norm_mean, self._norm_std),
            events,
            start,
            stop,
            weights,
            initial_point=initial_point,
            show_progress=show_progress,
            step_size=step_size,
        )

        self.hazards_ = pd.Series(hazards_, index=df.columns, name="coef") / self._norm_std
        self.variance_matrix_ = -inv(self._hessian_) / np.outer(self._norm_std, self._norm_std)
        self.standard_errors_ = self._compute_standard_errors(
            normalize(df, self._norm_mean, self._norm_std), events, start, stop, weights
        )
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_cumulative_hazard_ = self._compute_cumulative_baseline_hazard(df, events, start, stop, weights)
        self.baseline_survival_ = self._compute_baseline_survival()
        self.event_observed = events
        self.start_stop_and_events = pd.DataFrame({"event": events, "start": start, "stop": stop})
        self.weights = weights

        self._n_examples = df.shape[0]
        self._n_unique = df.index.unique().shape[0]
        return self

    def _check_values(self, df, events, start, stop):
        # check_for_overlapping_intervals(df) # this is currently too slow for production.
        check_nans_or_infs(df)
        check_low_var(df)
        check_complete_separation_low_variance(df, events, self.event_col)
        check_for_numeric_dtypes_or_raise(df)
        check_for_immediate_deaths(events, start, stop)
        check_for_instantaneous_events(start, stop)

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
        for (
            (stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W),
            _,
        ) in self._partition_by_strata(X, events, start, stop, weights):
            yield function(stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W, *args)

    def _compute_z_values(self):
        return self.hazards_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_confidence_intervals(self):
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        hazards = self.hazards_.values
        return pd.DataFrame(
            np.c_[hazards - z * se, hazards + z * se], columns=["lower-bound", "upper-bound"], index=self.hazards_.index
        )

    @property
    def summary(self):
        """
        Summary statistics describing the fit.

        Returns
        -------
        df: DataFrame
            contains columns coef, exp(coef), se(coef), z, p, lower, upper
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            df = pd.DataFrame(index=self.hazards_.index)
            df["coef"] = self.hazards_
            df["exp(coef)"] = np.exp(self.hazards_)
            df["se(coef)"] = self.standard_errors_
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -np.log2(df["p"])
            df["lower %g" % (1 - self.alpha)] = self.confidence_intervals_["lower-bound"]
            df["upper %g" % (1 - self.alpha)] = self.confidence_intervals_["upper-bound"]
            return df

    def _newton_rhaphson(
        self,
        df,
        events,
        start,
        stop,
        weights,
        show_progress=False,
        step_size=None,
        precision=10e-6,
        max_steps=50,
        initial_point=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        """
        Newton Rhaphson algorithm for fitting CPH model.

        Parameters
        ----------
        df: DataFrame
        stop_times_events: DataFrame
             meta information about the subjects history
        show_progress: boolean, optional (default: True)
            to show verbose output of convergence
        step_size: float
            > 0 to determine a starting step size in NR algorithm.
        precision: float
            the convergence halts if the norm of delta between
                     successive positions is less than epsilon.

        Returns
        --------
        beta: (1,d) numpy array.
        """
        assert precision <= 1.0, "precision must be less than or equal to 1."

        _, d = df.shape

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
                h, g, ll = self._get_gradients(
                    df.values, events.values, start.values, stop.values, weights.values, beta
                )
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

            if self.penalizer > 0:
                # add the gradient and hessian of the l2 term
                g -= self.penalizer * beta
                h.flat[:: d + 1] -= self.penalizer

            try:
                # reusing a piece to make g * inv(h) * g.T faster later
                inv_h_dot_g_T = spsolve(-h, g, sym_pos=True)
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
                    "\rIteration %d: norm_delta = %.5f, step_size = %.5f, ll = %.5f, newton_decrement = %.5f, seconds_since_start = %.1f"
                    % (i, norm_delta, step_size, ll, newton_decrement, time.time() - start_time),
                    end=""
                )

            # convergence criteria
            if norm_delta < precision:
                converging, completed = False, True
            elif previous_ll > 0 and abs(ll - previous_ll) / (-previous_ll) < 1e-09:
                # this is what R uses by default
                converging, completed = False, True
            elif newton_decrement < 10e-8:
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
See https://stats.stackexchange.com/questions/11109/how-to-deal-with-perfect-separation-in-logistic-regression",
                    ConvergenceWarning,
                )
                converging, completed = False, False

            step_size = step_sizer.update(norm_delta).next()

            beta += delta

        self._hessian_ = hessian
        self._score_ = gradient
        self._log_likelihood = ll

        if show_progress and completed:
            print("Convergence completed after %d iterations." % (i))
        elif show_progress and not completed:
            print("Convergence failed. See any warning messages.")

        # report to the user problems that we detect.
        if completed and norm_delta > 0.1:
            warnings.warn(
                "Newton-Rhapson convergence completed but norm(delta) is still high, %.3f. This may imply non-unique solutions to the maximum likelihood. Perhaps there is colinearity or complete separation in the dataset?"
                % norm_delta,
                ConvergenceWarning,
            )
        elif not completed:
            warnings.warn("Newton-Rhapson failed to converge sufficiently in %d steps." % max_steps, ConvergenceWarning)

        return beta

    def _get_gradients(self, X, events, start, stop, weights, beta):  # pylint: disable=too-many-locals
        """
        Calculates the first and second order vector differentials, with respect to beta.

        Returns
        -------
        hessian: (d, d) numpy array,
        gradient: (d,) numpy array
        log_likelihood: float
        """

        _, d = X.shape
        hessian = np.zeros((d, d))
        gradient = np.zeros(d)
        log_lik = 0
        # weights = weights[:, None]
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
            risk_phi_x = matrix_axis_0_sum_to_array(phi_x_i)
            risk_phi_x_x = phi_x_x_i

            # Calculate the sums of Tie set
            deaths = events_at_t & (stops_events_at_t == t)

            tied_death_counts = array_sum_to_scalar(deaths.astype(int))  # should always at least 1

            xi_deaths = X_at_t[deaths]

            x_death_sum = matrix_axis_0_sum_to_array(weights_at_t[deaths, None] * xi_deaths)

            weight_count = array_sum_to_scalar(weights_at_t[deaths])
            weighted_average = weight_count / tied_death_counts

            #
            # This code is near identical to the _batch algorithm in CoxPHFitter. In fact, see _batch for comments.
            #

            if tied_death_counts > 1:

                # A good explaination for how Efron handles ties. Consider three of five subjects who fail at the time.
                # As it is not known a priori that who is the first to fail, so one-third of
                # (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
                # of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

                # a lot of this is now in einstien notation for performance, but see original "expanded" code here
                # https://github.com/CamDavidsonPilon/lifelines/blob/e7056e7817272eb5dff5983556954f56c33301b1/lifelines/fitters/cox_time_varying_fitter.py#L458-L490

                tie_phi = array_sum_to_scalar(phi_i[deaths])
                tie_phi_x = matrix_axis_0_sum_to_array(phi_x_i[deaths])
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

    def predict_log_partial_hazard(self, X):
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - \bar{x})'\beta `


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
        if isinstance(X, pd.DataFrame):
            order = self.hazards_.index
            X = X[order]
            check_for_numeric_dtypes_or_raise(X)

        X = X.astype(float)
        index = _get_index(X)
        X = normalize(X, self._norm_mean.values, 1)
        return pd.DataFrame(np.dot(X, self.hazards_), index=index)

    def predict_partial_hazard(self, X):
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

    def print_summary(self, decimals=2, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        kwargs:
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        # Print information about data first
        justify = string_justify(18)

        print(self)
        print("{} = '{}'".format(justify("event col"), self.event_col))

        if self.weights_col:
            print("{} = '{}'".format(justify("weights col"), self.weights_col))

        if self.strata:
            print("{} = {}".format(justify("strata"), self.strata))

        if self.penalizer > 0:
            print("{} = {}".format(justify("penalizer"), self.penalizer))

        print("{} = {}".format(justify("number of subjects"), self._n_unique))
        print("{} = {}".format(justify("number of periods"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.{prec}f}".format(justify("log-likelihood"), self._log_likelihood, prec=decimals))
        print("{} = {} UTC".format(justify("time fit was run"), self._time_fit_was_called))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        # Significance codes last
        print(
            df.to_string(
                float_format=format_floats(decimals),
                formatters={"p": format_p_value(decimals), "exp(coef)": format_exp_floats(decimals)},
            )
        )

        # Significance code explanation
        print("---")
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = self.log_likelihood_ratio_test()
            print(
                "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                    sr.test_statistic, sr.degrees_freedom, -np.log2(sr.p_value), prec=decimals
                )
            )

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
                ._log_likelihood
            )

        ll_alt = self._log_likelihood
        test_stat = 2 * (ll_alt - ll_null)
        degrees_freedom = self.hazards_.shape[0]
        p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value,
            test_stat,
            name="log-likelihood ratio test",
            degrees_freedom=degrees_freedom,
            null_distribution="chi squared",
        )

    def plot(self, columns=None, **errorbar_kwargs):
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

        ax = errorbar_kwargs.pop("ax", None) or plt.figure().add_subplot(111)

        errorbar_kwargs.setdefault("c", "k")
        errorbar_kwargs.setdefault("fmt", "s")
        errorbar_kwargs.setdefault("markerfacecolor", "white")
        errorbar_kwargs.setdefault("markeredgewidth", 1.25)
        errorbar_kwargs.setdefault("elinewidth", 1.25)
        errorbar_kwargs.setdefault("capsize", 3)

        z = inv_normal_cdf(1 - self.alpha / 2)

        if columns is None:
            columns = self.hazards_.index

        yaxis_locations = list(range(len(columns)))
        symmetric_errors = z * self.standard_errors_[columns].to_frame().squeeze(axis=1).values.copy()
        hazards = self.hazards_[columns].values.copy()

        order = np.argsort(hazards)

        ax.errorbar(hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
        ax.set_ylim(best_ylim)

        tick_labels = [columns[i] for i in order]

        ax.set_yticks(yaxis_locations)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def _compute_cumulative_baseline_hazard(
        self, tv_data, events, start, stop, weights
    ):  # pylint: disable=too-many-locals
        hazards = self.predict_partial_hazard(tv_data).values

        unique_death_times = np.unique(stop[events.values])
        baseline_hazard_ = pd.DataFrame(
            np.zeros_like(unique_death_times), index=unique_death_times, columns=["baseline hazard"]
        )

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
        """ approximate change in betas as a result of excluding ith row"""

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
            se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.Series(se, index=self.hazards_.index, name="se")
