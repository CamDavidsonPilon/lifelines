# -*- coding: utf-8 -*-
from __future__ import annotations
from functools import partial, wraps
from inspect import getfullargspec
from datetime import datetime
from textwrap import dedent
import typing as t
import collections
import warnings
import sys

from numpy.linalg import inv, pinv
import numpy as np

from autograd import hessian, value_and_grad, elementwise_grad as egrad, grad
from autograd.differential_operators import make_jvp_reversemode
from autograd.misc import flatten
import autograd.numpy as anp

from scipy.optimize import minimize, root_scalar
from scipy.integrate import trapezoid
from scipy import stats

import pandas as pd

from lifelines.plotting import _plot_estimate, set_kwargs_drawstyle
from lifelines.utils.printer import Printer
from lifelines import exceptions
from lifelines import utils


__all__ = [
    "ParametericAFTRegressionFitter",
    "ParametricRegressionFitter",
    "ParametricUnivariateFitter",
    "RegressionFitter",
    "UnivariateFitter",
    "BaseFitter",
]


class BaseFitter:

    weights: np.ndarray
    event_observed: np.ndarray

    def __init__(self, alpha: float = 0.05, label: str = None):
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha parameter must be between 0 and 1.")
        self.alpha = alpha
        self._class_name = self.__class__.__name__
        self._label = label
        self._censoring_type = None

    def __repr__(self) -> str:
        classname = self._class_name
        if self._label:
            label_string = """"%s",""" % self._label
        else:
            label_string = ""
        try:
            s = """<lifelines.%s:%s fitted with %g total observations, %g %s-censored observations>""" % (
                classname,
                label_string,
                self.weights.sum(),
                self.weights.sum() - self.weights[self.event_observed > 0].sum(),
                utils.CensoringType.str_censoring_type(self),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

    @property
    def label(self):
        return self._label

    @utils.CensoringType.right_censoring
    def fit(*args, **kwargs):
        raise NotImplementedError()

    @utils.CensoringType.right_censoring
    def fit_right_censoring(self, *args, **kwargs):
        """Alias for ``fit``

        See Also
        ---------
        fit
        """
        return self.fit(*args, **kwargs)


class UnivariateFitter(BaseFitter):

    survival_function_: pd.DataFrame
    _estimate_name: str
    _estimation_method: str

    def plot(self, **kwargs):
        """
        Plots a pretty figure of the model

        Matplotlib plot arguments can be passed in inside the kwargs, plus

        Parameters
        -----------
        show_censors: bool
            place markers at censorship events. Default: False
        censor_styles: dict
            If show_censors, this dictionary will be passed into the plot call.
        ci_alpha: float
            the transparency level of the confidence interval. Default: 0.3
        ci_force_lines: bool
            force the confidence intervals to be line plots (versus default shaded areas). Default: False
        ci_show: bool
            show confidence intervals. Default: True
        ci_legend: bool
            if ci_force_lines is True, this is a boolean flag to add the lines' labels to the legend. Default: False
        at_risk_counts: bool
            show group sizes at time points. See function ``add_at_risk_counts`` for details. Default: False
        loc: slice
            specify a time-based subsection of the curves to plot, ex:

            >>> model.plot(loc=slice(0.,10.))

            will plot the time values between t=0. and t=10.
        iloc: slice
            specify a location-based subsection of the curves to plot, ex:

            >>> model.plot(iloc=slice(0,10))

            will plot the first 10 time points.

        Returns
        -------
        ax:
            a pyplot axis object
        """
        warnings.warn(
            "The `plot` function is deprecated, and will be removed in future versions. Use `plot_%s`" % self._estimate_name,
            DeprecationWarning,
        )
        # Fix the confidence interval plot bug from Aalen-Johansen
        # when calculate_variance is False.
        if getattr(self, "_calc_var", None) is False:
            kwargs["ci_show"] = False
        return _plot_estimate(self, estimate=self._estimate_name, **kwargs)

    def subtract(self, other) -> pd.DataFrame:
        """
        Subtract self's survival function from another model's survival function.

        Parameters
        ----------
        other: same object as self

        """
        self_estimate = getattr(self, self._estimate_name)
        other_estimate = getattr(other, other._estimate_name)
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        return pd.DataFrame(
            self_estimate.reindex(new_index, method="ffill").values - other_estimate.reindex(new_index, method="ffill").values,
            index=new_index,
            columns=["diff"],
        )

    def divide(self, other) -> pd.DataFrame:
        """
        Divide self's survival function from another model's survival function.

        Parameters
        ----------
        other: same object as self

        """
        self_estimate = getattr(self, self._estimate_name)
        other_estimate = getattr(other, other._estimate_name)
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        t = pd.DataFrame(
            self_estimate.reindex(new_index, method="ffill").values / other_estimate.reindex(new_index, method="ffill").values,
            index=new_index,
            columns=["ratio"],
        )
        return t

    def predict(self, times: t.Union[t.Iterable[float], float], interpolate=False) -> pd.Series:
        """
        Predict the fitter at certain point in time. Uses a linear interpolation if
        points in time are not in the index.

        Parameters
        ----------
        times: scalar, or array
            a scalar or an array of times to predict the value of {0} at.
        interpolate: bool, optional (default=False)
            for methods that produce a stepwise solution (Kaplan-Meier, Nelson-Aalen, etc), turning this to
            True will use an linear interpolation method to provide a more "smooth" answer.

        """
        if callable(self._estimation_method):
            return (
                pd.DataFrame(self._estimation_method(utils._to_1d_array(times)), index=utils._to_1d_array(times))
                .loc[times]
                .squeeze()
            )

        estimate = getattr(self, self._estimation_method)
        if not interpolate:
            return estimate.asof(times).squeeze()

        warnings.warn("Approximating using linear interpolation`.\n", exceptions.ApproximationWarning)
        return utils.interpolate_at_times_and_return_pandas(estimate, times)

    @property
    def conditional_time_to_event_(self) -> pd.DataFrame:
        """
        Return a DataFrame, with index equal to survival_function_, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        individual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.
        """
        age = self.survival_function_.index.values[:, None]
        columns = ["%s - Conditional median duration remaining to event" % self.label]
        return (
            pd.DataFrame(
                utils.qth_survival_times(self.survival_function_[self.label] * 0.5, self.survival_function_)
                .sort_index(ascending=False)
                .values,
                index=self.survival_function_.index,
                columns=columns,
            )
            - age
        )

    def hazard_at_times(self, times, label=None):
        raise NotImplementedError

    def survival_function_at_times(self, times, label=None):
        raise NotImplementedError

    def cumulative_hazard_at_times(self, times, label=None):
        raise NotImplementedError

    def cumulative_density_at_times(self, times, label=None):
        raise NotImplementedError

    def plot_cumulative_hazard(self, **kwargs):
        raise NotImplementedError()

    def plot_survival_function(self, **kwargs):
        raise NotImplementedError()

    def plot_hazard(self, **kwargs):
        raise NotImplementedError()

    def plot_cumulative_density(self, **kwargs):
        raise NotImplementedError()

    def plot_density(self, **kwargs):
        raise NotImplementedError()

    @property
    def median_survival_time_(self) -> float:
        """
        Return the unique time point, t, such that S(t) = 0.5. This is the "half-life" of the population, and a
        robust summary statistic for the population, if it exists.
        """
        return self.percentile(0.5)

    def percentile(self, p: float) -> float:
        """
        Return the unique time point, t, such that S(t) = p.

        Parameters
        -----------
        p: float
        """
        warnings.warn(
            "Approximating using `survival_function_`. To increase accuracy, try using or increasing the resolution of the timeline kwarg in `.fit(..., timeline=timeline)`.\n",
            exceptions.ApproximationWarning,
        )
        return utils.qth_survival_times(p, self.survival_function_)


class NonParametricUnivariateFitter(UnivariateFitter):
    pass


class ParametricUnivariateFitter(UnivariateFitter):
    """
    Without overriding anything, assumes all parameters must be greater than 0.
    """

    _KNOWN_MODEL = False
    _MIN_PARAMETER_VALUE = 1e-9
    _scipy_fit_method = "L-BFGS-B"
    _scipy_fit_options: dict[str, t.Any] = dict()
    _scipy_fit_callback = None
    _fitted_parameter_names: list[str]

    def __init__(self, *args, **kwargs):
        super(ParametricUnivariateFitter, self).__init__(*args, **kwargs)
        self._estimate_name = "cumulative_hazard_"
        if not hasattr(self, "_bounds"):
            self._bounds = [(0.0, None)] * len(self._fitted_parameter_names)
        self._bounds = list(self._buffer_bounds(self._bounds))

        if "alpha" in self._fitted_parameter_names:
            raise NameError("'alpha' in _fitted_parameter_names is a lifelines reserved word. Try 'alpha_' instead.")

    @property
    def params_(self):
        return self.summary["coef"]

    @property
    def AIC_(self) -> float:
        return -2 * self.log_likelihood_ + 2 * self._fitted_parameters_.shape[0]

    @property
    def BIC_(self) -> float:
        return -2 * self.log_likelihood_ + len(self._fitted_parameter_names) * np.log(self.event_observed.shape[0])

    def _check_cumulative_hazard_is_monotone_and_positive(self, durations, values):
        class_name = self._class_name

        cumulative_hazard = self._cumulative_hazard(values, durations)
        if not np.all(cumulative_hazard >= 0):
            warnings.warn(
                dedent(
                    """\
                Cumulative hazard is not strictly positive. For example, try:

                >>> fitter = {0}()
                >>> fitter._cumulative_hazard(np.{1}, np.sort(durations))

                This may harm convergence, or return nonsensical results.
            """.format(
                        class_name, values.__repr__()
                    )
                ),
                exceptions.StatisticalWarning,
            )

        derivative_of_cumulative_hazard = self._hazard(values, durations)
        if not np.all(derivative_of_cumulative_hazard >= 0):
            warnings.warn(
                dedent(
                    """\
                Cumulative hazard is not strictly non-decreasing. For example, try:

                >>> fitter = {0}()
                >>> # Recall: the hazard is the derivative of the cumulative hazard
                >>> fitter._hazard({1}, np.sort(durations))

                This may harm convergence, or return nonsensical results.
            """.format(
                        class_name, values.__repr__()
                    )
                ),
                exceptions.StatisticalWarning,
            )

    def _initial_values_from_bounds(self):
        for (lb, ub) in self._bounds:
            if lb is None and ub is None:
                yield 0.0
            elif lb is None:
                yield ub - 1.0
            elif ub is None:
                yield lb + 1.0
            else:
                yield (ub - lb) / 2.0

    def _buffer_bounds(self, bounds: list[tuple[t.Optional[float], t.Optional[float]]]):
        for (lb, ub) in bounds:
            if lb is None and ub is None:
                yield (None, None)
            elif lb is None and ub is not None:
                yield (None, ub - self._MIN_PARAMETER_VALUE)
            elif ub is None and lb is not None:
                yield (lb + self._MIN_PARAMETER_VALUE, None)
            elif ub is not None and lb is not None:
                yield (lb + self._MIN_PARAMETER_VALUE, ub - self._MIN_PARAMETER_VALUE)

    def _cumulative_hazard(self, params, times):
        return -anp.log(self._survival_function(params, times))

    def _hazard(self, *args, **kwargs):
        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        return egrad(self._cumulative_hazard, argnum=1)(*args, **kwargs)

    def _density(self, *args, **kwargs):
        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        return egrad(self._cumulative_density, argnum=1)(*args, **kwargs)

    def _survival_function(self, params, times):
        return anp.exp(-self._cumulative_hazard(params, times))

    def _cumulative_density(self, params, times):
        return 1 - self._survival_function(params, times)

    def _log_hazard(self, params, times):
        hz = self._hazard(params, times)
        hz = anp.clip(hz, 1e-50, np.inf)
        return anp.log(hz)

    def _log_1m_sf(self, params, times):
        # equal to log(cdf), but often easier to express with sf.
        return anp.log1p(-self._survival_function(params, times))

    def _negative_log_likelihood_left_censoring(self, params, Ts, E, entry, weights) -> float:
        T = Ts[1]
        non_zero_entries = entry > 0

        log_hz = self._log_hazard(params, T)
        cum_haz = self._cumulative_hazard(params, T)
        log_1m_sf = self._log_1m_sf(params, T)

        ll = (E * weights * (log_hz - cum_haz - log_1m_sf)).sum() + (weights * log_1m_sf).sum()
        ll = ll + (weights[non_zero_entries] * self._cumulative_hazard(params, entry[non_zero_entries])).sum()

        return -ll / weights.sum()

    def _negative_log_likelihood_right_censoring(self, params, Ts, E, entry, weights) -> float:
        T = Ts[0]
        non_zero_entries = entry > 0

        log_hz = self._log_hazard(params, T[E])
        cum_haz = self._cumulative_hazard(params, T)

        ll = (weights[E] * log_hz).sum() - (weights * cum_haz).sum()
        ll = ll + (weights[non_zero_entries] * self._cumulative_hazard(params, entry[non_zero_entries])).sum()
        return -ll / weights.sum()

    def _negative_log_likelihood_interval_censoring(self, params, Ts, E, entry, weights) -> float:
        start, stop = Ts
        non_zero_entries = entry > 0
        observed_weights, censored_weights = weights[E], weights[~E]
        censored_starts = start[~E]
        observed_stops, censored_stops = stop[E], stop[~E]

        ll = (observed_weights * self._log_hazard(params, observed_stops)).sum() - (
            observed_weights * self._cumulative_hazard(params, observed_stops)
        ).sum()

        # this diff can be 0 - we can't take the log of that.
        ll = (
            ll
            + (
                censored_weights
                * anp.log(
                    anp.clip(
                        self._survival_function(params, censored_starts) - self._survival_function(params, censored_stops),
                        1e-25,
                        1 - 1e-25,
                    )
                )
            ).sum()
        )
        ll = ll + (weights[non_zero_entries] * self._cumulative_hazard(params, entry[non_zero_entries])).sum()
        return -ll / weights.sum()

    def _compute_confidence_bounds_of_cumulative_hazard(self, alpha, ci_labels) -> pd.DataFrame:
        return self._compute_confidence_bounds_of_transform(self._cumulative_hazard, alpha, ci_labels, self.timeline)

    def _compute_variance_of_transform(self, transform, timeline=None):
        """
        This computes the variance of a transform of the parameters. Ex: take
        the fitted parameters, a function/transform and the variance matrix and give me
        back variance of the transform.

        Parameters
        -----------
        transform: function
            must a function of two parameters:
                ``params``, an iterable that stores the parameters
                ``times``, a numpy vector representing some timeline
            the function must use autograd imports (scipy and numpy)

        """
        if timeline is None:
            timeline = self.timeline
        else:
            timeline = utils._to_1d_array(timeline)

        # pylint: disable=no-value-for-parameter
        gradient_of_transform_at_mle = make_jvp_reversemode(transform)(self._fitted_parameters_, timeline.astype(float))

        gradient_at_times = np.vstack(
            [gradient_of_transform_at_mle(basis) for basis in np.eye(len(self._fitted_parameters_), dtype=float)]
        )

        return pd.Series(
            np.einsum("nj,jk,nk->n", gradient_at_times.T, self.variance_matrix_, gradient_at_times.T), index=timeline
        )

    def _compute_confidence_bounds_of_transform(
        self, transform, alpha: float, ci_labels: tuple[str, str], timeline
    ) -> pd.DataFrame:
        """
        This computes the confidence intervals of a transform of the parameters. Ex: take
        the fitted parameters, a function/transform and the variance matrix and give me
        back confidence intervals of the transform.

        Parameters
        -----------
        transform: function
            must a function of two parameters:
                ``params``, an iterable that stores the parameters
                ``times``, a numpy vector representing some timeline
            the function must use autograd imports (scipy and numpy)
        alpha: float
            confidence level
        ci_labels: tuple
        timeline: iterable

        """
        alpha2 = 1 - alpha / 2.0
        z = utils.inv_normal_cdf(alpha2)
        df = pd.DataFrame(index=timeline)

        std_of_transform = np.sqrt(self._compute_variance_of_transform(transform))

        if ci_labels is None:
            ci_labels = ["%s_lower_%g" % (self.label, 1 - alpha), "%s_upper_%g" % (self.label, 1 - alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = transform(self._fitted_parameters_, timeline) - z * std_of_transform
        df[ci_labels[1]] = transform(self._fitted_parameters_, timeline) + z * std_of_transform
        return df

    def _create_initial_point(self, *args) -> np.ndarray:
        # this can be overwritten in the model class.
        # *args has terms like Ts, E, entry, weights
        return np.array(list(self._initial_values_from_bounds()))

    def _fit_model(self, Ts, E, entry, weights, fit_options: dict, show_progress=True):

        if utils.CensoringType.is_left_censoring(self):
            negative_log_likelihood = self._negative_log_likelihood_left_censoring
        elif utils.CensoringType.is_interval_censoring(self):
            negative_log_likelihood = self._negative_log_likelihood_interval_censoring
        elif utils.CensoringType.is_right_censoring(self):
            negative_log_likelihood = self._negative_log_likelihood_right_censoring

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            minimizing_results, previous_results, minimizing_ll = None, None, np.inf
            for method, option in zip(
                ["Nelder-Mead", self._scipy_fit_method],
                [{"maxiter": 400}, {**{"disp": show_progress}, **self._scipy_fit_options, **fit_options}],
            ):

                initial_value = self._initial_values if previous_results is None else utils._to_1d_array(previous_results.x)

                results = minimize(
                    value_and_grad(negative_log_likelihood),  # pylint: disable=no-value-for-parameter
                    initial_value,
                    jac=True,
                    method=method,
                    args=(Ts, E, entry, weights),
                    bounds=self._bounds,
                    options=option,
                    callback=self._scipy_fit_callback,
                )
                previous_results = results

                if results.success and ~np.isnan(results.x).any() and (results.fun < minimizing_ll):
                    minimizing_ll = results.fun
                    minimizing_results = results

            # convergence successful.
            # I still need to check for ~np.isnan(minimizing_results.x).any() since minimize will happily
            # return nans even when criteria is satisfied.
            if minimizing_results and minimizing_results.success and ~np.isnan(minimizing_results.x).any():
                sol = utils._to_1d_array(minimizing_results.x)
                # pylint: disable=no-value-for-parameter
                hessian_ = hessian(negative_log_likelihood)(sol, Ts, E, entry, weights)
                # see issue https://github.com/CamDavidsonPilon/lifelines/issues/801
                hessian_ = (hessian_ + hessian_.T) / 2
                return sol, -minimizing_results.fun * weights.sum(), hessian_ * weights.sum()

            # convergence failed.
            if show_progress:
                print(minimizing_results)
            if self._KNOWN_MODEL:
                raise exceptions.ConvergenceError(
                    dedent(
                        """\
                    Fitting did not converge. This is mostly a lifelines problem, but a few things you can check:

                    1. Are there any extreme values in the durations column?
                      - Try scaling your durations to a more reasonable values closer to 1 (multiplying or dividing by some 10^n). If this works,
                        then likely you just need to specify good initial values with `initial_point` argument in the call to `fit`.
                      - Try dropping them to see if the model converges.
                    2. %s may just be a poor model of the data. Try another parametric model.
                """
                        % self._class_name
                    )
                )

            else:
                raise exceptions.ConvergenceError(
                    dedent(
                        """\
                    Fitting did not converge.

                    1. Are two parameters in the model collinear / exchangeable? (Change model)
                    2. Is the cumulative hazard always non-negative and always non-decreasing? (Assumption error)
                    3. Are there inputs to the cumulative hazard that could produce NaNs or Infs? (Check your _bounds)

                    This could be a problem with your data:
                    1. Are there any extreme values in the durations column?
                        - Try scaling your durations to a more reasonable value closer to 1 (multiplying or dividing by a large constant).
                        - Try dropping them to see if the model converges.
                    2. %s may just be a poor model of the data. Try another parametric model.

                    """
                        % self._class_name
                    )
                )

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _estimation_method(self, t):
        return self.survival_function_at_times(t)

    def _compute_standard_errors(self) -> pd.DataFrame:
        return pd.DataFrame(
            [np.sqrt(self.variance_matrix_.values.diagonal())], index=["se"], columns=self._fitted_parameter_names
        )

    def _compute_confidence_bounds_of_parameters(self) -> pd.DataFrame:
        se = self._compute_standard_errors().loc["se"]
        z = utils.inv_normal_cdf(1 - self.alpha / 2.0)
        return pd.DataFrame(
            np.c_[self._fitted_parameters_ - z * se, self._fitted_parameters_ + z * se],
            columns=["lower-bound", "upper-bound"],
            index=self._fitted_parameter_names,
        )

    def _compute_z_values(self):
        return (self._fitted_parameters_ - self._compare_to_values) / self._compute_standard_errors().loc["se"]

    @property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics describing the fit.

        See Also
        --------
        print_summary
        """
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            ci = (1 - self.alpha) * 100
            lower_upper_bounds = self._compute_confidence_bounds_of_parameters()
            df = pd.DataFrame(index=self._fitted_parameter_names)
            df["coef"] = self._fitted_parameters_
            df["se(coef)"] = self._compute_standard_errors().loc["se"]
            df["coef lower %g%%" % ci] = lower_upper_bounds["lower-bound"]
            df["coef upper %g%%" % ci] = lower_upper_bounds["upper-bound"]
            df["cmp to"] = self._compare_to_values
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -utils.quiet_log2(df["p"])
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
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        justify = utils.string_rjustify(25)

        p = Printer(
            self,
            [
                ("number of observations", "{:g}".format(self.weights.sum())),
                ("number of events observed", "{:g}".format(self.weights[self.event_observed > 0].sum())),
                ("log-likelihood", "{:.{prec}f}".format(self.log_likelihood_, prec=decimals)),
                (
                    "hypothesis",
                    ", ".join(
                        "%s != %g" % (name, iv) for (name, iv) in zip(self._fitted_parameter_names, self._compare_to_values)
                    ),
                ),
            ],
            [("AIC", "{:.{prec}f}".format(self.AIC_, prec=decimals))],
            justify,
            kwargs,
            decimals,
            columns,
        )

        p.print(style=style)

    @utils.CensoringType.right_censoring
    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        show_progress=False,
        entry=None,
        weights=None,
        initial_point=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricUnivariateFitter:  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
        durations: an array, or pd.Series
          length n, duration subject was observed for
        event_observed: numpy array or pd.Series, optional
          length n, True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
        timeline: list, optional
            return the estimate at the values in timeline (positively increasing)
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: list, optional
            add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
        show_progress: bool, optional
            since this is an iterative fitting algorithm, switching this to True will display some iteration details.
        entry: an array, or pd.Series, of length n
            relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
            entered study when they were "born": time zero.
        weights: an array, or pd.Series, of length n
            integer weights per observation
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
          self
            self with new properties like ``cumulative_hazard_``, ``survival_function_``

        """

        self.durations = np.asarray(utils.pass_for_numeric_dtypes_or_raise_array(durations))
        utils.check_nans_or_infs(self.durations)
        utils.check_positivity(self.durations)

        return self._fit(
            (self.durations, None),
            event_observed=event_observed,
            timeline=timeline,
            label=label,
            alpha=alpha,
            ci_labels=ci_labels,
            show_progress=show_progress,
            entry=entry,
            weights=weights,
            initial_point=initial_point,
            fit_options=fit_options,
        )

    @utils.CensoringType.left_censoring
    def fit_left_censoring(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        show_progress=False,
        entry=None,
        weights=None,
        initial_point=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricUnivariateFitter:  # pylint: disable=too-many-arguments
        """
        Fit the model to a left-censored dataset

        Parameters
        ----------
        durations: an array, or pd.Series
          length n, duration subject was observed for
        event_observed: numpy array or pd.Series, optional
          length n, True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
        timeline: list, optional
            return the estimate at the values in timeline (positively increasing)
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: list, optional
            add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
        show_progress: bool, optional
            since this is an iterative fitting algorithm, switching this to True will display some iteration details.
        entry: an array, or pd.Series, of length n
            relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
            entered study when they were "born": time zero.
        weights: an array, or pd.Series, of length n
            integer weights per observation
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
            self with new properties like ``cumulative_hazard_``, ``survival_function_``

        """

        self.durations = np.asarray(utils.pass_for_numeric_dtypes_or_raise_array(durations))
        utils.check_nans_or_infs(self.durations)
        utils.check_positivity(self.durations)
        return self._fit(
            (None, self.durations),
            event_observed=event_observed,
            timeline=timeline,
            label=label,
            alpha=alpha,
            ci_labels=ci_labels,
            show_progress=show_progress,
            entry=entry,
            weights=weights,
            initial_point=initial_point,
            fit_options=fit_options,
        )

    @utils.CensoringType.interval_censoring
    def fit_interval_censoring(
        self,
        lower_bound,
        upper_bound,
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        show_progress=False,
        entry=None,
        weights=None,
        initial_point=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricUnivariateFitter:  # pylint: disable=too-many-arguments
        """
        Fit the model to an interval censored dataset.

        Parameters
        ----------
        lower_bound: an array, or pd.Series
          length n, the start of the period the subject experienced the event in.
        upper_bound: an array, or pd.Series
          length n, the end of the period the subject experienced the event in. If the value is equal to the corresponding value in lower_bound, then
          the individual's event was observed (not censored).
        event_observed: numpy array or pd.Series, optional
          length n, if left optional, infer from ``lower_bound`` and ``upper_bound`` (if lower_bound==upper_bound then event observed, if lower_bound < upper_bound, then event censored)
        timeline: list, optional
            return the estimate at the values in timeline (positively increasing)
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: list, optional
            add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
        show_progress: bool, optional
            since this is an iterative fitting algorithm, switching this to True will display some iteration details.
        entry: an array, or pd.Series, of length n
            relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
            entered study when they were "born": time zero.
        weights: an array, or pd.Series, of length n
            integer weights per observation
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
          self
            self with new properties like ``cumulative_hazard_``, ``survival_function_``

        """
        self.upper_bound = np.atleast_1d(utils.pass_for_numeric_dtypes_or_raise_array(upper_bound))
        self.lower_bound = np.atleast_1d(utils.pass_for_numeric_dtypes_or_raise_array(lower_bound))

        utils.check_nans_or_infs(self.lower_bound)
        utils.check_positivity(self.upper_bound)

        if (self.upper_bound < self.lower_bound).any():
            raise ValueError("All upper_bound times must be greater than or equal to lower_bound times.")

        if event_observed is None:
            event_observed = self.upper_bound == self.lower_bound

        if ((self.lower_bound == self.upper_bound) != event_observed).any():
            raise ValueError(
                "For all rows, lower_bound == upper_bound if and only if event observed = 1 (uncensored). Likewise, lower_bound < upper_bound if and only if event observed = 0 (censored)"
            )

        return self._fit(
            (np.clip(self.lower_bound, 1e-20, 1e25), np.clip(self.upper_bound, 1e-20, 1e25)),
            event_observed=event_observed,
            timeline=timeline,
            label=label,
            alpha=alpha,
            ci_labels=ci_labels,
            show_progress=show_progress,
            entry=entry,
            weights=weights,
            initial_point=initial_point,
            fit_options=fit_options,
        )

    def _fit(
        self,
        Ts: tuple[t.Optional[np.ndarray], t.Optional[np.ndarray]],
        event_observed=None,
        timeline=None,
        label=None,
        alpha=None,
        ci_labels=None,
        show_progress=False,
        entry=None,
        weights=None,
        initial_point=None,
        fit_options=None,
    ) -> ParametricUnivariateFitter:

        n = len(utils.coalesce(*Ts))

        if event_observed is not None:
            event_observed = np.asarray(event_observed)
            utils.check_nans_or_infs(event_observed)

        self.event_observed = np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones(n)

        self.entry = np.asarray(entry) if entry is not None else np.zeros(n)
        self.weights = np.asarray(weights) if weights is not None else np.ones(n)

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline).astype(float))
        else:
            self.timeline = np.linspace(utils.coalesce(*Ts).min(), utils.coalesce(*Ts).max(), min(n, 500))

        self._label = utils.coalesce(label, self._label, self._class_name.replace("Fitter", "") + "_estimate")
        self._ci_labels = ci_labels
        self.alpha = utils.coalesce(alpha, self.alpha)
        fit_options = utils.coalesce(fit_options, dict())

        # create some initial values, and test them in the hazard.
        self._initial_values = utils.coalesce(
            initial_point, self._create_initial_point(Ts, self.event_observed, self.entry, self.weights)
        )
        self._check_bounds_initial_point_names_shape()

        if not hasattr(self, "_compare_to_values"):
            self._compare_to_values = self._initial_values

        if not self._KNOWN_MODEL:
            self._check_cumulative_hazard_is_monotone_and_positive(utils.coalesce(*Ts), self._initial_values)

        # estimation
        self._fitted_parameters_, self.log_likelihood_, self._hessian_ = self._fit_model(
            Ts, self.event_observed.astype(bool), self.entry, self.weights, show_progress=show_progress, fit_options=fit_options
        )

        if not self._KNOWN_MODEL:
            self._check_cumulative_hazard_is_monotone_and_positive(utils.coalesce(*Ts), self._fitted_parameters_)

        for param_name, fitted_value in zip(self._fitted_parameter_names, self._fitted_parameters_):
            setattr(self, param_name, fitted_value)

        try:
            variance_matrix_ = inv(self._hessian_)
        except np.linalg.LinAlgError:
            variance_matrix_ = pinv(self._hessian_)
            warning_text = dedent(
                """\

                The Hessian for %s's fit was not invertible. We will instead approximate it using the pseudo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the fitted parameters too. Perform plots of the cumulative hazard to help understand the latter's bias.
                """
                % self._class_name
            )
            warnings.warn(warning_text, exceptions.ApproximationWarning)
        finally:
            if (variance_matrix_.diagonal() < 0).any() or np.isnan(variance_matrix_).any():
                warning_text = dedent(
                    """\
                    The diagonal of the variance_matrix_ has negative values or NaNs. This could be a problem with %s's fit to the data.

                    It's advisable to not trust the variances reported, and to be suspicious of the fitted parameters too. Perform plots of the cumulative hazard to help understand the latter's bias.

                    To fix this, try specifying an `initial_point` kwarg in `fit`.
                    """
                    % self._class_name
                )
                warnings.warn(warning_text, exceptions.StatisticalWarning)

        self.variance_matrix_ = pd.DataFrame(
            variance_matrix_, index=self._fitted_parameter_names, columns=self._fitted_parameter_names
        )

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame()
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame()
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame()
        self.cumulative_density_ = self.cumulative_density_at_times(self.timeline).to_frame()
        self.density_ = self.density_at_times(self.timeline).to_frame()
        return self

    def _check_bounds_initial_point_names_shape(self):
        if len(self._bounds) != len(self._fitted_parameter_names) != self._initial_values.shape[0]:
            raise ValueError(
                "_bounds must be the same shape as _fitted_parameter_names must be the same shape as _initial_values.\n"
            )

    @property
    def event_table(self) -> t.Union[pd.DataFrame, None]:
        if hasattr(self, "_event_table"):
            return self._event_table
        else:
            if utils.CensoringType.is_right_censoring(self):
                self._event_table = utils.survival_table_from_events(
                    self.durations, self.event_observed, self.entry, weights=self.weights
                )
            else:
                self._event_table = None
            return self.event_table

    def survival_function_at_times(self, times, label: t.Optional[str] = None) -> pd.Series:
        """
        Return a Pandas series of the predicted survival value at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the survival function at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        """
        label = utils.coalesce(label, self.label)
        return pd.Series(self._survival_function(self._fitted_parameters_, times), index=utils._to_1d_array(times), name=label)

    def cumulative_density_at_times(self, times, label: t.Optional[str] = None) -> pd.Series:
        """
        Return a Pandas series of the predicted cumulative density function (1-survival function) at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the survival function at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        """
        label = utils.coalesce(label, self.label)
        return pd.Series(self._cumulative_density(self._fitted_parameters_, times), index=utils._to_1d_array(times), name=label)

    def density_at_times(self, times, label=None) -> pd.Series:
        """
        Return a Pandas series of the predicted probability density function, dCDF/dt, at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the survival function at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        """
        label = utils.coalesce(label, self.label)
        return pd.Series(self._density(self._fitted_parameters_, times), index=utils._to_1d_array(times), name=label)

    def cumulative_hazard_at_times(self, times, label: t.Optional[str] = None) -> pd.Series:
        """
        Return a Pandas series of the predicted cumulative hazard value at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the cumulative hazard at.
        label: string, optional
          Rename the series returned. Useful for plotting.
        """
        label = utils.coalesce(label, self.label)
        return pd.Series(self._cumulative_hazard(self._fitted_parameters_, times), index=utils._to_1d_array(times), name=label)

    def hazard_at_times(self, times, label: t.Optional[str] = None) -> pd.Series:
        """
        Return a Pandas series of the predicted hazard at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the hazard at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        """
        label = utils.coalesce(label, self.label)
        return pd.Series(self._hazard(self._fitted_parameters_, times), index=utils._to_1d_array(times), name=label)

    @property
    def confidence_interval_(self) -> pd.DataFrame:
        """
        The confidence interval of the cumulative hazard. This is an alias for ``confidence_interval_cumulative_hazard_``.
        """
        return self._compute_confidence_bounds_of_cumulative_hazard(self.alpha, self._ci_labels)

    @property
    def confidence_interval_cumulative_hazard_(self) -> pd.DataFrame:
        """
        The confidence interval of the cumulative hazard. This is an alias for ``confidence_interval_``.
        """
        return self.confidence_interval_

    @property
    def confidence_interval_hazard_(self) -> pd.DataFrame:
        """
        The confidence interval of the hazard.
        """
        return self._compute_confidence_bounds_of_transform(self._hazard, self.alpha, self._ci_labels, self.timeline)

    @property
    def confidence_interval_density_(self) -> pd.DataFrame:
        """
        The confidence interval of the hazard.
        """
        return self._compute_confidence_bounds_of_transform(self._density, self.alpha, self._ci_labels, self.timeline)

    @property
    def confidence_interval_survival_function_(self) -> pd.DataFrame:
        """
        The lower and upper confidence intervals for the survival function
        """
        return self._compute_confidence_bounds_of_transform(self._survival_function, self.alpha, self._ci_labels, self.timeline)

    @property
    def confidence_interval_cumulative_density_(self) -> pd.DataFrame:
        """
        The lower and upper confidence intervals for the cumulative density
        """
        return self._compute_confidence_bounds_of_transform(self._cumulative_density, self.alpha, self._ci_labels, self.timeline)

    def plot(self, **kwargs):
        """
        Produce a pretty-plot of the estimate.
        """
        set_kwargs_drawstyle(kwargs, "default")
        warnings.warn(
            "The `plot` function is deprecated, and will be removed in future versions. Use `plot_%s`" % self._estimate_name,
            DeprecationWarning,
        )
        return _plot_estimate(self, estimate=self._estimate_name, **kwargs)

    def plot_cumulative_hazard(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(self, estimate="cumulative_hazard_", **kwargs)

    def plot_survival_function(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(self, estimate="survival_function_", **kwargs)

    def plot_cumulative_density(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(self, estimate="cumulative_density_", **kwargs)

    def plot_density(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(self, estimate="density_", **kwargs)

    def plot_hazard(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(self, estimate="hazard_", **kwargs)

    def _conditional_time_to_event_(self) -> pd.DataFrame:
        """
        Return a DataFrame, with index equal to survival_function_, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        individual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.

        Returns
        -------
        conditional_time_to_: DataFrame
            with index equal to survival_function_'s index

        """
        age = self.timeline
        columns = ["%s - Conditional median duration remaining to event" % self.label]

        return pd.DataFrame(self.percentile(0.5 * self.survival_function_.values) - age[:, None], index=age, columns=columns)

    def percentile(self, p: float) -> float:
        """
        Return the unique time point, t, such that S(t) = p.

        Parameters
        -----------
        p: float
        """
        # use numerical solver to find the value p = e^{-H(t)}. I think I could use `root` in scipy
        # instead of the scalar version. TODO
        def _find_root(_p):
            f = lambda t: _p - self.survival_function_at_times(t).values
            fprime = lambda t: self.survival_function_at_times(t).values * self.hazard_at_times(t).values
            return root_scalar(f, bracket=(1e-10, 2 * self.timeline[-1]), fprime=fprime, x0=1.0).root

        try:
            find_root = np.vectorize(_find_root, otypes=[float])
            return find_root(p)
        except ValueError:
            warnings.warn(
                "Looking like the model does not hit %g in the specified timeline. Try refitting with a larger timeline." % p,
                exceptions.StatisticalWarning,
            )
            return None


class KnownModelParametricUnivariateFitter(ParametricUnivariateFitter):

    _KNOWN_MODEL = True


class RegressionFitter(BaseFitter):

    _KNOWN_MODEL = False
    _FAST_MEDIAN_PREDICT = False
    _ALLOWED_RESIDUALS = {"schoenfeld", "score", "delta_beta", "deviance", "martingale", "scaled_schoenfeld"}

    def __init__(self, *args, **kwargs):
        super(RegressionFitter, self).__init__(*args, **kwargs)

    def plot_covariate_groups(self, *args, **kwargs):
        """
        Deprecated as of v0.25.0. Use ``plot_partial_effects_on_outcome`` instead.
        """
        warnings.warn("This method name is deprecated. Use `plot_partial_effects_on_outcome` instead.", DeprecationWarning)
        return self.plot_partial_effects_on_outcome(*args, **kwargs)

    def _compute_central_values_of_raw_training_data(self, df, strata=None, name="baseline"):
        """
        Compute our "baseline" observation for function like plot_partial_effects_on_outcome.
        - Categorical are transformed to their mode value.
        - Numerics are transformed to their median value.
        """
        if df.size == 0:
            return pd.DataFrame(index=["baseline"])

        if strata is not None:
            # apply this function within each stratified dataframe
            central_stats = []
            for stratum, df_ in df.groupby(strata):
                central_stats_ = self._compute_central_values_of_raw_training_data(df_, name=stratum)
                try:
                    central_stats_ = central_stats_.drop(strata, axis=1)
                except:
                    pass
                central_stats.append(central_stats_)
            v = pd.concat(central_stats)
            v.index.rename(utils.make_simpliest_hashable(strata), inplace=True)
            return v

        else:
            if pd.__version__ >= "1.1.0" and pd.__version__ < "2.0.0":
                # silence deprecation warning
                describe_kwarg = {"datetime_is_numeric": True}
            else:
                describe_kwarg = {}
            described = df.describe(include="all", **describe_kwarg)
            if "top" in described.index and "50%" not in described.index:
                central_stats = described.loc["top"].copy()
            elif "50%" in described.index and "top" not in described.index:
                central_stats = described.loc["50%"].copy()
            elif "top" in described.index and "50%" in described.index:
                central_stats = described.loc["top"].copy()
                central_stats.update(described.loc["50%"])

            central_stats = central_stats.to_frame(name=name).T.astype(df.dtypes)
            return central_stats

    def compute_residuals(self, training_dataframe: pd.DataFrame, kind: str) -> pd.DataFrame:
        """
        Compute the residuals the model.

        Parameters
        ----------
        training_dataframe : DataFrame
            the same training DataFrame given in `fit`
        kind : string
            One of {'schoenfeld', 'score', 'delta_beta', 'deviance', 'martingale', 'scaled_schoenfeld'}

        Notes
        -------
        - ``'scaled_schoenfeld'``: *lifelines* does not add the coefficients to the final results, but R does when you call ``residuals(c, "scaledsch")``



        """
        assert kind in self._ALLOWED_RESIDUALS, "kind must be in %s" % self._ALLOWED_RESIDUALS
        if self.entry_col is not None:
            raise NotImplementedError("Residuals for entries not implemented.")

        warnings.filterwarnings("ignore", category=exceptions.ConvergenceWarning)
        X, Ts, E, weights, _, shuffled_original_index, _ = self._preprocess_dataframe(training_dataframe)

        resids = getattr(self, "_compute_%s" % kind)(X, Ts, E, weights, index=shuffled_original_index)
        return resids


class SemiParametricRegressionFitter(RegressionFitter):
    @property
    def AIC_partial_(self) -> float:
        """
        "partial" because the log-likelihood is partial
        """
        return -2 * self.log_likelihood_ + 2 * self.params_.shape[0]


class ParametricRegressionFitter(RegressionFitter):

    _scipy_fit_method = "BFGS"
    _scipy_fit_options: dict[str, t.Any] = dict()
    _scipy_fit_callback = None
    fit_intercept = False
    force_no_intercept = False
    regressors = None
    strata = None

    def __init__(self, alpha: float = 0.05, penalizer: t.Union[float, np.ndarray] = 0.0, l1_ratio: float = 0.0, **kwargs):
        super(ParametricRegressionFitter, self).__init__(alpha=alpha, **kwargs)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def _check_values_post_fitting(self, df, T, E, weights, entries):
        utils.check_dimensions(df)
        utils.check_complete_separation(df, E, T, self.event_col)
        utils.check_scaling(df)

    def _pre_fit_model(self, Ts, E, Xs) -> None:
        return

    def _check_values_pre_fitting(self, df, T, E, weights, entries):
        utils.check_for_numeric_dtypes_or_raise(df)
        utils.check_nans_or_infs(df)
        utils.check_nans_or_infs(T)
        utils.check_nans_or_infs(E)
        utils.check_positivity(T)

        if self.weights_col:
            if (weights.astype(int) != weights).any() and not self.robust:
                warnings.warn(
                    dedent(
                        """It appears your weights are not integers, possibly propensity or sampling scores then?
                                        It's important to know that the naive variance estimates of the coefficients are biased. Instead a) set `robust=True` in the call to `fit`, or b) use Monte Carlo to
                                        estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"""
                    ),
                    exceptions.StatisticalWarning,
                )
            if (weights <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

        if self.entry_col:
            utils.check_entry_times(T, entries)

    def _cumulative_hazard(self, params, T, Xs):
        return -anp.log(self._survival_function(params, T, Xs))

    def _hazard(self, params, T, Xs):
        return egrad(self._cumulative_hazard, argnum=1)(params, T, Xs)  # pylint: disable=unexpected-keyword-arg

    def _log_hazard(self, params, T, Xs):
        # can be overwritten to improve convergence, see example in WeibullAFTFitter
        hz = self._hazard(params, T, Xs)
        hz = anp.clip(hz, 1e-20, np.inf)
        return anp.log(hz)

    def _log_1m_sf(self, params, T, Xs):
        # equal to log(cdf), but often easier to express with sf.
        return anp.log1p(-self._survival_function(params, T, Xs))

    def _survival_function(self, params, T, Xs):
        return anp.clip(anp.exp(-self._cumulative_hazard(params, T, Xs)), 1e-12, 1 - 1e-12)

    def _log_likelihood_right_censoring(self, params, Ts: tuple, E, W, entries, Xs) -> float:

        T = Ts[0]
        non_zero_entries = entries > 0

        log_hz = self._log_hazard(params, T, Xs)
        cum_hz = self._cumulative_hazard(params, T, Xs)
        delayed_entries = self._cumulative_hazard(params, entries[non_zero_entries], Xs.filter(non_zero_entries))

        ll = 0
        ll = ll + (W * E * log_hz).sum()
        ll = ll + -(W * cum_hz).sum()
        ll = ll + (W[non_zero_entries] * delayed_entries).sum()
        ll = ll / anp.sum(W)
        return ll

    def _log_likelihood_left_censoring(self, params, Ts, E, W, entries, Xs) -> float:

        T = Ts[1]
        non_zero_entries = entries > 0

        log_hz = self._log_hazard(params, T, Xs)
        cum_haz = self._cumulative_hazard(params, T, Xs)
        log_1m_sf = self._log_1m_sf(params, T, Xs)
        delayed_entries = self._cumulative_hazard(params, entries[non_zero_entries], Xs.filter(non_zero_entries))

        ll = 0
        ll = (W * E * (log_hz - cum_haz - log_1m_sf)).sum() + (W * log_1m_sf).sum()
        ll = ll + (W[non_zero_entries] * delayed_entries).sum()
        ll = ll / anp.sum(W)
        return ll

    def _log_likelihood_interval_censoring(self, params, Ts, E, W, entries, Xs) -> float:

        start, stop = Ts
        non_zero_entries = entries > 0
        observed_deaths = self._log_hazard(params, stop[E], Xs.filter(E)) - self._cumulative_hazard(params, stop[E], Xs.filter(E))
        censored_interval_deaths = anp.log(
            anp.clip(
                self._survival_function(params, start[~E], Xs.filter(~E))
                - self._survival_function(params, stop[~E], Xs.filter(~E)),
                1e-25,
                1 - 1e-25,
            )
        )
        delayed_entries = self._cumulative_hazard(params, entries[non_zero_entries], Xs.filter(non_zero_entries))

        ll = 0
        ll = ll + (W[E] * observed_deaths).sum()
        ll = ll + (W[~E] * censored_interval_deaths).sum()
        ll = ll + (W[non_zero_entries] * delayed_entries).sum()
        ll = ll / anp.sum(W)
        return ll

    @utils.CensoringType.left_censoring
    def fit_left_censoring(
        self,
        df,
        duration_col=None,
        event_col=None,
        regressors=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricRegressionFitter:
        """
        Fit the regression model to a left-censored dataset.

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
            lifetimes/measurements/etc. This column contains the (possibly) left-censored data.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        regressors: dict, optional
            a dictionary of parameter names -> {list of column names, formula} that maps model parameters
            to a linear combination of variables. If left as None, all variables
            will be used for all parameters.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: str
            specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
            self with additional new properties ``print_summary``, ``params_``, ``confidence_intervals_`` and more

        """
        self.duration_col = duration_col

        df = df.copy()

        T = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(duration_col)).astype(float)
        self.durations = T.copy()

        self._fit(
            self._log_likelihood_left_censoring,
            df,
            (None, T.values),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
            fit_options=fit_options,
        )

        return self

    @utils.CensoringType.interval_censoring
    def fit_interval_censoring(
        self,
        df,
        lower_bound_col,
        upper_bound_col,
        event_col=None,
        ancillary=None,
        regressors=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricRegressionFitter:
        """
        Fit the regression model to a interval-censored dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights).
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

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        regressors: dict, optional
            a dictionary of parameter names -> {list of column names, formula} that maps model parameters
            to a linear combination of variables. If left as None, all variables
            will be used for all parameters.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: string
            specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
            self with additional new properties: ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        """
        df = df.copy()

        self.upper_bound_col = upper_bound_col
        self.lower_bound_col = lower_bound_col

        self.lower_bound = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(lower_bound_col)).astype(float)
        self.upper_bound = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(upper_bound_col)).astype(float)

        if event_col is None:
            event_col = "E_lifelines_added"
            df[event_col] = self.lower_bound == self.upper_bound

        if ((self.lower_bound == self.upper_bound) != df[event_col]).any():
            raise ValueError(
                "For all rows, lower_bound == upper_bound if and only if event observed = 1 (uncensored). Likewise, lower_bound < upper_bound if and only if event observed = 0 (censored)"
            )
        if (self.lower_bound > self.upper_bound).any():
            raise ValueError("All upper bound measurements must be greater than or equal to lower bound measurements.")

        self._fit(
            self._log_likelihood_interval_censoring,
            df,
            (self.lower_bound.values, np.clip(self.upper_bound.values, 0, 1e25)),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
            fit_options=fit_options,
        )

        return self

    @utils.CensoringType.right_censoring
    def fit(
        self,
        df,
        duration_col,
        event_col=None,
        regressors=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricRegressionFitter:
        """
        Fit the regression model to a right-censored dataset.

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

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        regressors: dict, optional
            a dictionary of parameter names -> {list of column names, formula} that maps model parameters
            to a linear combination of variables. If left as None, all variables
            will be used for all parameters.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: string
            specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
            self with additional new properties: ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        """
        self.duration_col = duration_col

        df = df.copy()

        self.durations = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(duration_col)).astype(float)

        self._fit(
            self._log_likelihood_right_censoring,
            df,
            (self.durations.values, None),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
            fit_options=fit_options,
        )

        return self

    def _fit(
        self,
        log_likelihood_function,
        df,
        Ts,
        regressors,
        event_col=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        fit_options: Optional[dict] = None,
    ) -> ParametricRegressionFitter:

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self._n_examples = df.shape[0]
        self.weights_col = weights_col
        self.entry_col = entry_col
        self.event_col = event_col
        self.robust = robust

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline).astype(float))
        else:
            self.timeline = np.unique(utils.coalesce(*Ts))

        E = (
            utils.pass_for_numeric_dtypes_or_raise_array(df.pop(self.event_col))
            if (self.event_col is not None)
            else pd.Series(np.ones(self._n_examples, dtype=bool), index=df.index, name="E")
        )
        weights = (
            utils.pass_for_numeric_dtypes_or_raise_array(df.pop(self.weights_col)).astype(float)
            if (self.weights_col is not None)
            else pd.Series(np.ones(self._n_examples, dtype=float), index=df.index, name="weights")
        )

        entries = (
            utils.pass_for_numeric_dtypes_or_raise_array(df.pop(self.entry_col)).astype(float)
            if (self.entry_col is not None)
            else pd.Series(np.zeros(self._n_examples, dtype=float), index=df.index, name="entry")
        )

        utils.check_nans_or_infs(E)
        E = E.astype(bool)
        self.event_observed = E.copy()
        self.entry = entries.copy()
        self.weights = weights.copy()
        self._central_values = self._compute_central_values_of_raw_training_data(df, self.strata)

        regressors = utils.coalesce(regressors, self.regressors, {p: None for p in self._fitted_parameter_names})

        self.regressors = utils.CovariateParameterMappings(
            regressors, df, force_intercept=self.fit_intercept, force_no_intercept=self.force_no_intercept
        )
        Xs = self.regressors.transform_df(df)

        self._check_values_pre_fitting(Xs, utils.coalesce(Ts[1], Ts[0]), E, weights, entries)

        _norm_std = Xs.std(0)
        _index = Xs.columns

        self._cols_to_not_penalize = self._find_cols_to_not_penalize(_norm_std)
        self._norm_std = Xs.std(0)
        _constant_cols = pd.Series(
            [_norm_std.loc[param_name, variable_name] < 1e-8 for (param_name, variable_name) in _index], index=_index
        )
        self._norm_std[_constant_cols] = 1.0
        _norm_std[_norm_std < 1e-8] = 1.0

        self._pre_fit_model(Ts, E, Xs)

        _params, self.log_likelihood_, self._hessian_ = self._fit_model(
            log_likelihood_function,
            Ts,
            utils.normalize(Xs, 0, _norm_std),
            E.values,
            weights.values,
            entries.values,
            fit_options=utils.coalesce(fit_options, dict()),
            show_progress=show_progress,
            user_supplied_initial_point=initial_point,
        )

        # assert the coefficients are aligned.
        # https://github.com/CamDavidsonPilon/lifelines/issues/931
        assert list(self.regressors.keys()) == list(self._norm_std.index.get_level_values(0).unique())
        _params = np.concatenate([_params[k] for k in self.regressors.keys()])

        self.params_ = _params / self._norm_std
        self._compare_to_values = np.zeros_like(self.params_)
        self.variance_matrix_ = pd.DataFrame(self._compute_variance_matrix(), index=_index, columns=_index)
        self.standard_errors_ = self._compute_standard_errors(Ts, E.values, weights.values, entries.values, Xs)
        self.confidence_intervals_ = self._compute_confidence_intervals()
        if self._FAST_MEDIAN_PREDICT:
            self._predicted_median = self.predict_median(df)
        return self

    def _find_cols_to_not_penalize(self, norm_std):
        """
        We only want to avoid penalizing the constant term in linear relationships. Our flag for a
        linear relationship is >1 covariate.
        """
        index = norm_std.index
        s = pd.Series(False, index=index)
        for k, v in index.groupby(index.get_level_values(0)).items():
            if v.size > 1:
                for (parameter_name, variable_name) in v:
                    if norm_std.loc[parameter_name, variable_name] < 1e-8:
                        s.loc[(parameter_name, variable_name)] = True

        return s

    def _create_initial_point(self, Ts, E, entries, weights, Xs) -> t.Union[list[dict], dict]:
        return {parameter_name: np.zeros(len(Xs[parameter_name].columns)) for parameter_name in self._fitted_parameter_names}

    def _add_penalty(self, params: dict, neg_ll: float):
        params_array, _ = flatten(params)

        # remove intercepts from being penalized
        params_array = params_array[~self._cols_to_not_penalize]
        if (isinstance(self.penalizer, np.ndarray) or self.penalizer > 0) and self.l1_ratio > 0:
            penalty = (
                self.l1_ratio * (self.penalizer * anp.abs(params_array)).sum()
                + 0.5 * (1.0 - self.l1_ratio) * (self.penalizer * (params_array) ** 2).sum()
            )

        elif (isinstance(self.penalizer, np.ndarray) or self.penalizer > 0) and self.l1_ratio <= 0:
            penalty = 0.5 * (self.penalizer * (params_array) ** 2).sum()

        else:
            penalty = 0
        return neg_ll + penalty

    def _create_neg_likelihood_with_penalty_function(
        self, params_array, Ts, E, weights, entries, Xs, likelihood=None, penalty=None
    ):
        # it's a bit unfortunate but we do have to "flatten" each time this is called.
        # I've tried making it an attribute and freezing it with partial, but both caused serialization issues.
        _, unflatten_array_to_dict = flatten(self._initial_point_dicts[0])
        params_dict = unflatten_array_to_dict(params_array)
        if penalty is None:
            return -likelihood(params_dict, Ts, E, weights, entries, Xs)
        else:
            return penalty(params_dict, -likelihood(params_dict, Ts, E, weights, entries, Xs))

    def _prepare_initial_points(self, user_supplied_initial_point, Ts, E, entries, weights, Xs):
        self._initial_point_dicts = utils._to_list(self._create_initial_point(Ts, E, entries, weights, Xs))
        _, unflatten = flatten(self._initial_point_dicts[0])

        if user_supplied_initial_point is not None and isinstance(user_supplied_initial_point, dict):
            initial_point_arrays, _ = flatten(user_supplied_initial_point)
            initial_point_arrays = [initial_point_arrays]
        elif user_supplied_initial_point is not None and isinstance(user_supplied_initial_point, np.ndarray):
            initial_point_arrays = [user_supplied_initial_point]
        elif user_supplied_initial_point is None:
            # not supplied by user
            initial_point_arrays = [flatten(initial_point_dict)[0] for initial_point_dict in self._initial_point_dicts]
        return initial_point_arrays, unflatten

    def _fit_model(
        self, likelihood, Ts, Xs, E, weights, entries, fit_options, show_progress=False, user_supplied_initial_point=None
    ):
        initial_points_as_arrays, unflatten_array_to_dict = self._prepare_initial_points(
            user_supplied_initial_point, Ts, E, entries, weights, Xs
        )

        # optimizing this function
        self._neg_likelihood_with_penalty_function = partial(
            self._create_neg_likelihood_with_penalty_function, likelihood=likelihood, penalty=self._add_penalty
        )

        # scoring this function in `score`
        self._neg_likelihood = partial(self._create_neg_likelihood_with_penalty_function, likelihood=likelihood)

        minimum_ll = np.inf
        minimum_results = None
        for _initial_point in initial_points_as_arrays:

            if _initial_point.shape[0] != Xs.columns.size:
                raise ValueError("initial_point is not the correct shape.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = minimize(
                    # using value_and_grad is much faster (takes advantage of shared computations) than splitting.
                    value_and_grad(self._neg_likelihood_with_penalty_function),
                    _initial_point,
                    method=self._scipy_fit_method,
                    jac=True,
                    args=(Ts, E, weights, entries, utils.DataframeSlicer(Xs)),
                    options={**{"disp": show_progress}, **self._scipy_fit_options, **fit_options},
                    callback=self._scipy_fit_callback,
                )

            if results.fun < minimum_ll:
                minimum_ll, minimum_results = results.fun, results

        if show_progress:
            print(minimum_results)

        if minimum_results is not None and minimum_results.success:
            sum_weights = weights.sum()
            hessian_ = hessian(self._neg_likelihood_with_penalty_function)(
                minimum_results.x, Ts, E, weights, entries, utils.DataframeSlicer(Xs)
            )
            # See issue https://github.com/CamDavidsonPilon/lifelines/issues/801
            hessian_ = (hessian_ + hessian_.T) / 2
            return (unflatten_array_to_dict(minimum_results.x), -sum_weights * minimum_results.fun, sum_weights * hessian_)
        else:
            self._check_values_post_fitting(Xs, utils.coalesce(Ts[1], Ts[0]), E, weights, entries)
            raise exceptions.ConvergenceError(
                dedent(
                    f"""\

                Fitting did not converge. Try the following:

                0. Are there any lifelines warnings outputted during the `fit`?
                1. Inspect your DataFrame: does everything look as expected?
                2. Try scaling your duration vector down, i.e. `df[duration_col] = df[duration_col]/100`
                3. Is there high-collinearity in the dataset? Try using the variance inflation factor (VIF) to find redundant variables.
                4. Try using an alternate minimizer: ``fitter._scipy_fit_method = "SLSQP"``.
                5. Trying adding a small penalizer (or changing it, if already present). Example: `{self._class_name}(penalizer=0.01).fit(...)`.
                6. Are there any extreme outliers? Try modeling them or dropping them to see if it helps convergence.

                minimum_results={minimum_results}
            """
                )
            )

    def score(self, df: pd.DataFrame, scoring_method: str = "log_likelihood") -> float:
        """
        Score the data in df on the fitted model. With default scoring method, returns
        the _average log-likelihood_.

        Parameters
        ----------
        df: DataFrame
            the dataframe with duration col, event col, etc.
        scoring_method: str
            one of {'log_likelihood', 'concordance_index'}
            log_likelihood: returns the average unpenalized log-likelihood.
            concordance_index: returns the concordance-index

        Examples
        ---------
        .. code:: python

            from lifelines import WeibullAFTFitter
            from lifelines.datasets import load_rossi

            rossi_train = load_rossi().loc[:400]
            rossi_test = load_rossi().loc[400:]
            wf = WeibullAFTFitter().fit(rossi_train, 'week', 'arrest')

            wf.score(rossi_train)
            wf.score(rossi_test)
        """
        df = df.copy()
        if scoring_method == "log_likelihood":
            if utils.CensoringType.is_left_censoring(self):
                Ts = (None, df.pop(self.duration_col).values)
                E = df.pop(self.event_col).astype(bool).values
            elif utils.CensoringType.is_interval_censoring(self):
                Ts = (df.pop(self.lower_bound_col).values, df.pop(self.upper_bound_col).values)
                E = Ts[0] == Ts[1]
            elif utils.CensoringType.is_right_censoring(self):
                Ts = (df.pop(self.duration_col).values, None)
                E = df.pop(self.event_col).astype(bool).values

            if self.weights_col:
                try:
                    W = df.pop(self.weights_col).values
                except:
                    W = np.ones_like(E, dtype=float)
            else:
                W = np.ones_like(E, dtype=float)

            if self.entry_col:
                entries = df.pop(self.entry_col).values
            else:
                entries = np.zeros_like(E, dtype=float)

            if self.strata:
                df = df.set_index(self.strata)

            Xs = self.regressors.transform_df(df)

            return -self._neg_likelihood(self.params_.values, Ts, E, W, entries, utils.DataframeSlicer(Xs))

        elif scoring_method == "concordance_index":
            T = df.pop(self.duration_col).values
            E = df.pop(self.event_col).values
            predictions = self.predict_median(df)

            return utils.concordance_index(T, predictions, E)
        else:
            raise NotImplementedError()

    def _compute_variance_matrix(self) -> np.ndarray:
        try:
            unit_scaled_variance_matrix_ = np.linalg.inv(self._hessian_)
        except np.linalg.LinAlgError:
            unit_scaled_variance_matrix_ = np.linalg.pinv(self._hessian_)
            warning_text = dedent(
                """\
                The Hessian was not invertible. We will instead approximate it using the pseudo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the fitted parameters too.

                Some ways to possible ways fix this:

                0. Are there any lifelines warnings outputted during the `fit`?
                1. Does a particularly large variable need to be centered to 0?
                2. Inspect your DataFrame: does everything look as expected? Do you need to add/drop a constant (intercept) column?
                3. Is there high-collinearity in the dataset? Try using the variance inflation factor (VIF) to find redundant variables.
                4. Trying adding a small penalizer (or changing it, if already present). Example: `%s(penalizer=0.01).fit(...)`.
                5. Are there any extreme outliers? Try modeling them or dropping them to see if it helps convergence.
                """
                % self._class_name
            )
            warnings.warn(warning_text, exceptions.ApproximationWarning)
        finally:
            if (unit_scaled_variance_matrix_.diagonal() < 0).any():
                warning_text = dedent(
                    """\
                    The diagonal of the variance_matrix_ has negative values. This could be a problem with %s's fit to the data.

                    It's advisable to not trust the variances reported, and to be suspicious of the fitted parameters too.
                    """
                    % self._class_name
                )
                warnings.warn(warning_text, exceptions.StatisticalWarning)

        return unit_scaled_variance_matrix_ / np.outer(self._norm_std, self._norm_std)

    def _compute_z_values(self):
        return (self.params_ - self._compare_to_values) / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_standard_errors(self, Ts, E, weights, entries, Xs) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.robust:
                se = np.sqrt(self._compute_sandwich_errors(Ts, E, weights, entries, Xs).diagonal())
            else:
                se = np.sqrt(self.variance_matrix_.values.diagonal())
            return pd.Series(se, name="se", index=self.params_.index)

    def _compute_sandwich_errors(self, Ts, E, weights, entries, Xs):
        with np.errstate(all="ignore"):
            # convergence will fail catastrophically elsewhere.

            ll_gradient = grad(self._neg_likelihood_with_penalty_function)
            params = self.params_.values
            n_params = params.shape[0]
            J = np.zeros((n_params, n_params))

            for ts, e, w, s, (_, xs) in zip(utils.safe_zip(*Ts), E, weights, entries, Xs.iterrows()):
                xs = utils.DataframeSlicer(xs.to_frame().T)
                score_vector = ll_gradient(params, ts, e, w, s, xs)
                J += np.outer(score_vector, score_vector)

            return self.variance_matrix_.values @ J @ self.variance_matrix_.values

    def _compute_confidence_intervals(self) -> pd.DataFrame:
        z = utils.inv_normal_cdf(1 - self.alpha / 2)
        ci = (1 - self.alpha) * 100
        se = self.standard_errors_
        params = self.params_.values
        return pd.DataFrame(
            np.c_[params - z * se, params + z * se],
            index=self.params_.index,
            columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
        )

    @property
    def _ll_null_dof(self) -> int:
        return len(self._fitted_parameter_names)

    @property
    def _ll_null(self) -> float:
        if hasattr(self, "_ll_null_"):
            return self._ll_null_

        regressors = {name: "1" for name in self._fitted_parameter_names}

        # we can reuse the final values from the full fit for this partial fit.
        initial_point = {}
        for name in self._fitted_parameter_names:
            try:
                initial_point[name] = self.params_[name]["Intercept"]
            except:
                initial_point[name] = 0.0001

        df = pd.DataFrame({"entry": self.entry, "w": self.weights})

        # some fitters will have custom __init__ fields that need to be provided (Piecewise, Spline...)
        args_to_provide = {k: getattr(self, k) for k in getfullargspec(self.__class__.__init__).args if k != "self"}
        args_to_provide["penalizer"] = self.penalizer
        model = self.__class__(**args_to_provide)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if utils.CensoringType.is_right_censoring(self):
                df["T"], df["E"] = self.durations, self.event_observed
                model.fit_right_censoring(
                    df, "T", "E", entry_col="entry", weights_col="w", regressors=regressors, initial_point=initial_point
                )
            elif utils.CensoringType.is_interval_censoring(self):
                df["lb"], df["ub"], df["E"] = self.lower_bound, self.upper_bound, self.event_observed
                model.fit_interval_censoring(
                    df, "lb", "ub", "E", entry_col="entry", weights_col="w", regressors=regressors, initial_point=initial_point
                )
            if utils.CensoringType.is_left_censoring(self):
                df["T"], df["E"] = self.durations, self.event_observed
                model.fit_left_censoring(
                    df, "T", "E", entry_col="entry", weights_col="w", regressors=regressors, initial_point=initial_point
                )
        self._ll_null_ = model.log_likelihood_
        return self._ll_null_

    def log_likelihood_ratio_test(self) -> "StatisticalResult":
        """
        This function computes the likelihood ratio test for the model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.
        """
        from lifelines.statistics import _chisq_test_p_value, StatisticalResult

        ll_null = self._ll_null
        ll_alt = self.log_likelihood_
        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.params_.shape[0] - self._ll_null_dof  # delta in number of parameters between models
        p_value = _chisq_test_p_value(test_stat, degrees_freedom=degrees_freedom)
        return StatisticalResult(
            p_value,
            test_stat,
            test_name="log-likelihood ratio test",
            degrees_freedom=degrees_freedom,
            null_distribution="chi squared",
        )

    @property
    def summary(self) -> pd.DataFrame:
        """
        Summary statistics describing the fit.

        See Also
        --------
        print_summary
        """

        ci = (1 - self.alpha) * 100
        z = utils.inv_normal_cdf(1 - self.alpha / 2)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = np.exp(self.params_)
            df["se(coef)"] = self.standard_errors_
            df["coef lower %g%%" % ci] = self.confidence_intervals_["%g%% lower-bound" % ci]
            df["coef upper %g%%" % ci] = self.confidence_intervals_["%g%% upper-bound" % ci]
            df["exp(coef) lower %g%%" % ci] = np.exp(self.params_) * np.exp(-z * self.standard_errors_)
            df["exp(coef) upper %g%%" % ci] = np.exp(self.params_) * np.exp(z * self.standard_errors_)
            df["cmp to"] = self._compare_to_values
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -utils.quiet_log2(df["p"])
            return df

    def print_summary(self, decimals: int = 2, style: t.Optional[str] = None, columns: t.Optional[list] = None, **kwargs) -> None:
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
        justify = utils.string_rjustify(25)
        headers: list[t.Any] = []

        if utils.CensoringType.is_interval_censoring(self):
            headers.extend(
                [("lower bound col", "'%s'" % self.lower_bound_col), ("upper bound col", "'%s'" % self.upper_bound_col)]
            )

        else:
            headers.append(("duration col", "'%s'" % self.duration_col))

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if self.entry_col:
            headers.append(("entry col", "'%s'" % self.entry_col))
        if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
            headers.append(("penalizer", self.penalizer))
        if self.robust:
            headers.append(("robust variance", True))

        headers.extend(
            [
                ("number of observations", "{:g}".format(self.weights.sum())),
                ("number of events observed", "{:g}".format(self.weights[self.event_observed > 0].sum())),
                ("log-likelihood", "{:.{prec}f}".format(self.log_likelihood_, prec=decimals)),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        sr = self.log_likelihood_ratio_test()
        footers = []

        if utils.CensoringType.is_right_censoring(self) and self._FAST_MEDIAN_PREDICT:
            footers.append(("Concordance", "{:.{prec}f}".format(self.concordance_index_, prec=decimals)))

        footers.extend(
            [
                ("AIC", "{:.{prec}f}".format(self.AIC_, prec=decimals)),
                (
                    "log-likelihood ratio test",
                    "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                ),
                ("-log2(p) of ll-ratio test", "{:.{prec}f}".format(-utils.quiet_log2(sr.p_value), prec=decimals)),
            ]
        )

        p = Printer(self, headers, footers, justify, kwargs, decimals, columns)

        p.print(style=style)

    def predict_survival_function(self, df, times=None, conditional_after=None) -> pd.DataFrame:
        """
        Predict the survival function for individuals, given their covariates. This assumes that the individual
        just entered the study (that is, we do not condition on how long they have already lived for.)

        Parameters
        ----------

        df: DataFrame
            a (n,d) DataFrame. If a DataFrame, columns
            can be in any order.
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved). Uses a linear interpolation if
            points in time are not in the index.
        conditional_after: iterable, optional
            Must be equal is size to df.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.


        Returns
        -------
        survival_function : DataFrame
            the survival probabilities of individuals over the timeline
        """
        return np.exp(-self.predict_cumulative_hazard(df, times=times, conditional_after=conditional_after))

    def predict_median(self, df, *, conditional_after=None) -> pd.DataFrame:
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order.
        conditional_after: iterable, optional
            Must be equal is size to df.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.

        Returns
        -------
        percentiles: DataFrame
            the median lifetimes for the individuals. If the survival curve of an
            individual does not cross 0.5, then the result is infinity.


        See Also
        --------
        predict_percentile, predict_expectation

        """
        return self.predict_percentile(df, p=0.5, conditional_after=conditional_after)

    def predict_percentile(self, df, *, p=0.5, conditional_after=None) -> pd.Series:
        if isinstance(df, pd.Series):
            df = df.to_frame().infer_objects().T
        subjects = utils._get_index(df)

        warnings.warn(
            "Approximating using `predict_survival_function`. To increase accuracy, try using or increasing the resolution of the timeline kwarg in `.fit(..., timeline=timeline)`.\n",
            exceptions.ApproximationWarning,
        )
        return utils.qth_survival_times(
            p, self.predict_survival_function(df, conditional_after=conditional_after)[subjects]
        ).T.squeeze()

    def predict_cumulative_hazard(self, df, *, times=None, conditional_after=None):
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

        times = utils.coalesce(times, self.timeline)
        times = np.atleast_1d(times).astype(float)

        n = df.shape[0]
        Xs = utils.DataframeSlicer(self.regressors.transform_df(df))

        params_dict = {parameter_name: self.params_.loc[parameter_name].values for parameter_name in self._fitted_parameter_names}

        columns = utils._get_index(df)
        if conditional_after is None:
            times_to_evaluate_at = np.tile(times, (n, 1)).T
            return pd.DataFrame(self._cumulative_hazard(params_dict, times_to_evaluate_at, Xs), index=times, columns=columns)
        else:
            conditional_after = np.asarray(conditional_after)
            times_to_evaluate_at = (conditional_after.reshape((n, 1)) + np.tile(times, (n, 1))).T
            return pd.DataFrame(
                np.clip(
                    self._cumulative_hazard(params_dict, times_to_evaluate_at, Xs)
                    - self._cumulative_hazard(params_dict, conditional_after.reshape(n), Xs),
                    0,
                    np.inf,
                ),
                index=times,
                columns=columns,
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
        times = utils.coalesce(times, self.timeline)
        times = np.atleast_1d(times).astype(float)

        n = df.shape[0]
        Xs = self.regressors.transform_df(df)

        params_dict = {parameter_name: self.params_.loc[parameter_name].values for parameter_name in self._fitted_parameter_names}

        if conditional_after is None:
            return pd.DataFrame(self._hazard(params_dict, np.tile(times, (n, 1)).T, Xs), index=times, columns=df.index)
        else:
            conditional_after = np.asarray(conditional_after)
            times_to_evaluate_at = (conditional_after.reshape((n, 1)) + np.tile(times, (n, 1))).T
            return pd.DataFrame(self._hazard(params_dict, times_to_evaluate_at, Xs), index=times, columns=df.index)

    def predict_expectation(self, X, conditional_after=None) -> pd.Series:
        r"""
        Compute the expected lifetime, :math:`E[T]`, using covariates X. This algorithm to compute the expectation is
        to use the fact that :math:`E[T] = \int_0^\inf P(T > t) dt = \int_0^\inf S(t) dt`. To compute the integral, we use the trapizoidal rule to approximate the integral.

        Caution
        --------
        If the survival function doesn't converge to 0, the the expectation is really infinity and the returned
        values are meaningless/too large. In that case, using ``predict_median`` or ``predict_percentile`` would be better.

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order.

        Returns
        -------
        expectations : DataFrame

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
        warnings.warn("""Approximating the expected value using trapezoid rule.\n""", exceptions.ApproximationWarning)
        subjects = utils._get_index(X)
        v = self.predict_survival_function(X, conditional_after=conditional_after)[subjects]
        return pd.Series(trapezoid(v.values.T, v.index), index=subjects).squeeze()

    @property
    def median_survival_time_(self):
        """
        The median survival time of the average subject in the training dataset.
        """
        return self.predict_median(self._central_values).squeeze()

    @property
    def mean_survival_time_(self):
        """
        The mean survival time of the average subject in the training dataset.
        """
        return self.predict_expectation(self._central_values).squeeze()

    def plot(self, columns=None, parameter=None, ax=None, **errorbar_kwargs):
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

        z = utils.inv_normal_cdf(1 - self.alpha / 2)

        params_ = self.params_.copy()
        standard_errors_ = self.standard_errors_.copy()
        user_supplied_columns = False

        if columns is not None:
            params_ = params_.loc[:, columns]
            standard_errors_ = standard_errors_.loc[:, columns]
            user_supplied_columns = True
        if parameter is not None:
            params_ = params_.loc[parameter]
            standard_errors_ = standard_errors_.loc[parameter]

        columns = params_.index

        hazards = params_.loc[columns].to_frame(name="coefs")
        hazards["se"] = z * standard_errors_.loc[columns]

        if not user_supplied_columns:
            if isinstance(hazards.index, pd.MultiIndex):
                hazards = hazards.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by="coefs", ascending=True))
            else:
                hazards = hazards.sort_values(by="coefs", ascending=True)

        yaxis_locations = list(range(len(columns)))

        ax.errorbar(hazards["coefs"], yaxis_locations, xerr=hazards["se"], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
        ax.set_ylim(best_ylim)

        if isinstance(columns[0], tuple):
            tick_labels = ["%s: %s" % (c, p) for (p, c) in hazards.index]
        else:
            tick_labels = [i for i in hazards.index]

        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("coef (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def plot_partial_effects_on_outcome(
        self, covariates, values, plot_baseline=True, ax=None, times=None, y="survival_function", **kwargs
    ):
        """
        Produces a plot comparing the baseline curve of the model versus
        what happens when a covariate(s) is varied over values in a group. This is useful to compare
        subjects' as we vary covariate(s), all else being held equal. The baseline
        curve is equal to the predicted y-curve at all average values in the original dataset.

        Parameters
        ----------
        covariates: string or list
            a string (or list of strings) of the covariate in the original dataset that we wish to vary.
        values: 1d or 2d iterable
            an iterable of the values we wish the covariate to take on.
        plot_baseline: bool
            also display the baseline survival, defined as the survival at the mean of the original dataset.
        times:
            pass in a times to plot
        y: str
            one of "survival_function", "hazard", "cumulative_hazard". Default "survival_function"
        kwargs:
            pass in additional plotting commands

        Returns
        -------
        ax: matplotlib axis, or list of axis'
            the matplotlib axis that be edited.

        Examples
        ---------
        .. code:: python

            from lifelines import datasets, WeibullAFTFitter
            rossi = datasets.load_rossi()
            wf = WeibullAFTFitter().fit(rossi, 'week', 'arrest')
            wf.plot_partial_effects_on_outcome('prio', values=np.arange(0, 15, 3), cmap='coolwarm')

        .. image:: /images/plot_covariate_example3.png

        .. code:: python

            # multiple variables at once
            wf.plot_partial_effects_on_outcome(['prio', 'paro'], values=[[0, 0], [5, 0], [10, 0], [0, 1], [5, 1], [10, 1]], cmap='coolwarm')

            # if you have categorical variables, you can simply things:
            wf.plot_partial_effects_on_outcome(['dummy1', 'dummy2', 'dummy3'], values=np.eye(3))

        """
        from matplotlib import pyplot as plt

        covariates = utils._to_list(covariates)
        values = np.atleast_1d(values)
        if len(values.shape) == 1:
            values = values[None, :].T

        if len(covariates) != values.shape[1]:
            raise ValueError("The number of covariates must equal to second dimension of the values array.")

        original_columns = self._central_values.columns
        for covariate in covariates:
            if covariate not in original_columns:
                raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        if ax is None:
            ax = plt.gca()

        # model X
        x_bar = self._central_values
        X = pd.concat([x_bar] * values.shape[0])
        if np.array_equal(np.eye(len(covariates)), values):
            X.index = ["%s=1" % c for c in covariates]
        else:
            X.index = [", ".join("%s=%s" % (c, v) for (c, v) in zip(covariates, row)) for row in values]
        for covariate, value in zip(covariates, values.T):
            X[covariate] = value

        getattr(self, "predict_%s" % y)(X, times=times).plot(ax=ax, **kwargs)
        if plot_baseline:
            getattr(self, "predict_%s" % y)(x_bar, times=times).rename(columns={0: "baseline survival"}).plot(
                ax=ax, ls=":", color="k"
            )
        return ax

    @property
    def concordance_index_(self) -> float:
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the ROC AUC
        to survival data, including censorships.
        For this purpose, the ``concordance_index_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset.
        """
        # pylint: disable=access-member-before-definition
        if not hasattr(self, "_concordance_index_"):
            try:
                self._concordance_index_ = utils.concordance_index(self.durations, self._predicted_median, self.event_observed)
                del self._predicted_median
                return self.concordance_index_
            except ZeroDivisionError:
                # this can happen if there are no observations, see #1172
                return 0.5
        return self._concordance_index_

    @property
    def AIC_(self) -> float:
        return -2 * self.log_likelihood_ + 2 * self.params_.shape[0]

    @property
    def BIC_(self) -> float:
        return -2 * self.log_likelihood_ + len(self._fitted_parameter_names) * np.log(self.event_observed.shape[0])


class ParametericAFTRegressionFitter(ParametricRegressionFitter):

    _KNOWN_MODEL = True
    _FAST_MEDIAN_PREDICT = True
    _primary_parameter_name: str
    _ancillary_parameter_name: str

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio=0.0, fit_intercept=True, model_ancillary=False):
        super(ParametericAFTRegressionFitter, self).__init__(alpha=alpha)
        # self._hazard = egrad(self._cumulative_hazard, argnum=1)  # pylint: disable=unexpected-keyword-arg
        self._fitted_parameter_names = [self._primary_parameter_name, self._ancillary_parameter_name]
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.model_ancillary = model_ancillary

    @utils.CensoringType.right_censoring
    def fit(
        self,
        df,
        duration_col,
        event_col=None,
        ancillary=None,
        fit_intercept=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        formula: str = None,
        fit_options: Optional[dict] = None,
    ) -> ParametericAFTRegressionFitter:
        """
        Fit the accelerated failure time model to a right-censored dataset.

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

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        formula: string
            Use an R-style formula for modeling the dataset. See formula syntax: https://matthewwardrop.github.io/formulaic/basic/grammar/
            If a formula is not provided, all variables in the dataframe are used (minus those used for other purposes like event_col, etc.)


        ancillary: None, boolean, str, or DataFrame, optional (default=None)
            Choose to model the ancillary parameters.
            If None or False, explicitly do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.
            If str, should be a formula

        fit_intercept: bool, optional
            If true, add a constant column to the regression. Overrides value set in class instantiation.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: string
            specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
             self with additional new properties ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        Examples
        --------

        .. code:: python

            from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            aft = WeibullAFTFitter()
            aft.fit(df, 'T', 'E')
            aft.print_summary()
            aft.predict_median(df)

            aft = WeibullAFTFitter()
            aft.fit(df, 'T', 'E', ancillary=df)
            aft.print_summary()
            aft.predict_median(df)

        """
        self.duration_col = duration_col
        self.fit_intercept = utils.coalesce(fit_intercept, self.fit_intercept)
        self.event_col = event_col
        self.entry_col = entry_col
        self.weights_col = weights_col

        df = df.copy()

        T = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(self.duration_col)).astype(float)
        self.durations = T.copy()

        if formula:
            primary_columns_or_formula = formula
        else:
            primary_columns_or_formula = df.columns.difference(
                [self.duration_col, self.event_col, self.entry_col, self.weights_col]
            ).tolist()

        regressors = {self._primary_parameter_name: primary_columns_or_formula}

        if isinstance(ancillary, pd.DataFrame):
            self.model_ancillary = True
            assert ancillary.shape[0] == df.shape[0], "ancillary must be the same shape[0] as df"
            regressors[self._ancillary_parameter_name] = ancillary.columns.difference(
                [self.duration_col, self.event_col, self.entry_col, self.weights_col]
            ).tolist()

            ancillary_cols_to_consider = ancillary.columns.difference(df.columns).difference([self.duration_col, self.event_col])
            df = pd.concat([df, ancillary[ancillary_cols_to_consider]], axis=1)

        elif isinstance(ancillary, str):
            # R-like formula
            self.model_ancillary = True
            regressors[self._ancillary_parameter_name] = ancillary

        elif (ancillary is True) or self.model_ancillary:
            self.model_ancillary = True
            regressors[self._ancillary_parameter_name] = regressors[self._primary_parameter_name]

        elif (ancillary is None) or (ancillary is False):
            regressors[self._ancillary_parameter_name] = "1"

        super(ParametericAFTRegressionFitter, self)._fit(
            self._log_likelihood_right_censoring,
            df,
            (T.values, None),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
            fit_options=fit_options,
        )
        return self

    @utils.CensoringType.interval_censoring
    def fit_interval_censoring(
        self,
        df,
        lower_bound_col,
        upper_bound_col,
        event_col=None,
        ancillary=None,
        fit_intercept=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        formula=None,
        fit_options: Optional[dict] = None,
    ) -> ParametericAFTRegressionFitter:
        """
        Fit the accelerated failure time model to a interval-censored dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns ``lower_bound_col``, ``upper_bound_col``  (see below),
            and any other covariates or weights.

        lower_bound_col: string
            the name of the column in DataFrame that contains the subjects'
            left-most observation.

        upper_bound_col: string
            the name of the column in DataFrame that contains the subjects'
            right-most observation. Values can be np.inf (and should be if the subject is right-censored).

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, will be inferred from the start and stop columns (lower_bound==upper_bound means uncensored)

        formula: string
            Use an R-style formula for modeling the dataset. See formula syntax: https://matthewwardrop.github.io/formulaic/basic/grammar/
            If a formula is not provided, all variables in the dataframe are used (minus those used for other purposes like event_col, etc.)

        ancillary: None, boolean, str, or DataFrame, optional (default=None)
            Choose to model the ancillary parameters.
            If None or False, explicitly do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.
            If str, should be a formula

        fit_intercept: bool, optional
            If true, add a constant column to the regression. Overrides value set in class instantiation.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: str
            specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
            self with additional new properties ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        Examples
        --------

        .. code:: python

            from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

            df = pd.DataFrame({
                'start': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'stop':  [5, 3, 9, 8, 7, 4, 8, 5, 2, 5, 6, np.inf],  # this last subject is right-censored.
                'E':     [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            aft = WeibullAFTFitter()
            aft.fit_interval_censoring(df, 'start', 'stop', 'E')
            aft.print_summary()
            aft.predict_median(df)

            aft = WeibullAFTFitter()
            aft.fit_interval_censoring(df, 'start', 'stop', 'E', ancillary=df)
            aft.print_summary()
            aft.predict_median(df)
        """

        self.lower_bound_col = lower_bound_col
        self.upper_bound_col = upper_bound_col
        self.fit_intercept = utils.coalesce(fit_intercept, self.fit_intercept)
        self.entry_col = entry_col
        self.weights_col = weights_col

        df = df.copy()

        lower_bound = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(lower_bound_col)).astype(float)
        upper_bound = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(upper_bound_col)).astype(float)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if event_col is None:
            event_col = "E_lifelines_added"
            df[event_col] = self.lower_bound == self.upper_bound

        self.event_col = event_col

        if ((self.lower_bound == self.upper_bound) != df[event_col]).any():
            raise ValueError(
                "For all rows, lower_bound == upper_bound if and only if event observed = 1 (uncensored). Likewise, lower_bound < upper_bound if and only if event observed = 0 (censored)"
            )
        if (self.lower_bound > self.upper_bound).any():
            raise ValueError("All upper bound measurements must be greater than or equal to lower bound measurements.")

        if formula:
            primary_columns_or_formula = formula
        else:
            primary_columns_or_formula = df.columns.difference(
                [self.lower_bound_col, self.upper_bound_col, self.event_col, self.entry_col, self.weights_col]
            ).tolist()

        regressors = {self._primary_parameter_name: primary_columns_or_formula}

        if isinstance(ancillary, pd.DataFrame):
            self.model_ancillary = True
            assert ancillary.shape[0] == df.shape[0], "ancillary must be the same shape[0] as df"
            regressors[self._ancillary_parameter_name] = ancillary.columns.difference(
                [self.lower_bound_col, self.upper_bound_col, self.event_col, self.entry_col, self.weights_col]
            ).tolist()

            ancillary_cols_to_consider = ancillary.columns.difference(df.columns).difference(
                [self.lower_bound_col, self.upper_bound_col, self.event_col]
            )
            df = pd.concat([df, ancillary[ancillary_cols_to_consider]], axis=1)

        elif isinstance(ancillary, str):
            # formula
            self.model_ancillary = True
            regressors[self._ancillary_parameter_name] = ancillary

        elif (ancillary is True) or self.model_ancillary:
            self.model_ancillary = True
            regressors[self._ancillary_parameter_name] = regressors[self._primary_parameter_name]

        elif (ancillary is None) or (ancillary is False):
            regressors[self._ancillary_parameter_name] = "1"

        super(ParametericAFTRegressionFitter, self)._fit(
            self._log_likelihood_interval_censoring,
            df,
            (lower_bound.values, np.clip(upper_bound.values, 0, 1e25)),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
            fit_options=fit_options,
        )
        return self

    @utils.CensoringType.left_censoring
    def fit_left_censoring(
        self,
        df,
        duration_col: str = None,
        event_col: str = None,
        ancillary=None,
        fit_intercept=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
        initial_point=None,
        entry_col=None,
        formula: str = None,
        fit_options: Optional[dict] = None,
    ) -> ParametericAFTRegressionFitter:
        """
        Fit the accelerated failure time model to a left-censored dataset.

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
            lifetimes/measurements/etc. This column contains the (possibly) left-censored data.

        event_col: string, optional
            the  name of the column in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        formula: string
            Use an R-style formula for modeling the dataset. See formula syntax: https://matthewwardrop.github.io/formulaic/basic/grammar/
            If a formula is not provided, all variables in the dataframe are used (minus those used for other purposes like event_col, etc.)

        ancillary: None, boolean, str, or DataFrame, optional (default=None)
            Choose to model the ancillary parameters.
            If None or False, explicitly do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.
            If str, should be a formula

        fit_intercept: bool, optional
            If true, add a constant column to the regression. Overrides value set in class instantiation.

        show_progress: bool, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in DataFrame that specifies weights per observation.

        robust: bool, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        entry_col: str
            specify a column in the DataFrame that denotes any late-entries (left truncation) that occurred. See
            the docs on `left truncation <https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#left-truncated-late-entry-data>`__

        fit_options: dict, optional
            pass kwargs into the underlying minimization algorithm, like ``tol``, etc.

        Returns
        -------
            self: self with additional new properties ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        Examples
        --------

        .. code:: python

            from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

            df = pd.DataFrame({
                'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
                'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
            })

            aft = WeibullAFTFitter()
            aft.fit_left_censoring(df, 'T', 'E')
            aft.print_summary()
            aft.predict_median(df)

            aft = WeibullAFTFitter()
            aft.fit_left_censoring(df, 'T', 'E', ancillary=df)
            aft.print_summary()
            aft.predict_median(df)
        """
        df = df.copy()

        T = utils.pass_for_numeric_dtypes_or_raise_array(df.pop(duration_col)).astype(float)
        self.duration_col = duration_col
        self.fit_intercept = utils.coalesce(fit_intercept, self.fit_intercept)
        self.event_col = event_col
        self.entry_col = entry_col
        self.weights_col = weights_col

        self.durations = T.copy()

        if formula:
            primary_columns_or_formula = formula
        else:
            primary_columns_or_formula = df.columns.difference(
                [self.duration_col, self.event_col, self.entry_col, self.weights_col]
            ).tolist()

        regressors = {self._primary_parameter_name: primary_columns_or_formula}

        if isinstance(ancillary, pd.DataFrame):
            self.model_ancillary = True
            assert ancillary.shape[0] == df.shape[0], "ancillary must be the same shape[0] as df"
            regressors[self._ancillary_parameter_name] = ancillary.columns.difference(
                [self.duration_col, self.event_col, self.entry_col, self.weights_col]
            ).tolist()

            ancillary_cols_to_consider = ancillary.columns.difference(df.columns).difference([self.duration_col, self.event_col])
            df = pd.concat([df, ancillary[ancillary_cols_to_consider]], axis=1)

        elif isinstance(ancillary, str):
            # R-like formula
            self.model_ancillary = True
            regressors[self._ancillary_parameter_name] = ancillary

        elif (ancillary is True) or self.model_ancillary:
            self.model_ancillary = True
            regressors[self._ancillary_parameter_name] = regressors[self._primary_parameter_name]

        elif (ancillary is None) or (ancillary is False):
            regressors[self._ancillary_parameter_name] = "1"

        super(ParametericAFTRegressionFitter, self)._fit(
            self._log_likelihood_left_censoring,
            df,
            (None, T.values),
            event_col=event_col,
            regressors=regressors,
            show_progress=show_progress,
            timeline=timeline,
            weights_col=weights_col,
            robust=robust,
            initial_point=initial_point,
            entry_col=entry_col,
            fit_options=fit_options,
        )

        return self

    def _create_initial_point(self, Ts, E, entries, weights, Xs):
        """
        See https://github.com/CamDavidsonPilon/lifelines/issues/664
        """
        constant_col = (Xs.std(0) < 1e-8).idxmax()

        def _transform_ith_param(param):
            if param <= 0:
                return param
            # technically this is suboptimal for log normal mu, but that's okay.
            return np.log(param)

        import lifelines  # kinda hacky but lol

        name = self._class_name.replace("AFT", "")
        try:
            uni_model = getattr(lifelines, name)()
        except AttributeError:
            # some custom AFT model if univariate model is not defined.
            return super(ParametericAFTRegressionFitter, self)._create_initial_point(Ts, E, entries, weights, Xs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if utils.CensoringType.is_right_censoring(self):
                uni_model.fit_right_censoring(Ts[0], event_observed=E, entry=entries, weights=weights)
            elif utils.CensoringType.is_interval_censoring(self):
                uni_model.fit_interval_censoring(Ts[0], Ts[1], entry=entries, weights=weights)
            elif utils.CensoringType.is_left_censoring(self):
                uni_model.fit_left_censoring(Ts[1], event_observed=E, entry=entries, weights=weights)

        # we may use this later in print_summary
        self._ll_null_ = uni_model.log_likelihood_

        initial_point = {}
        cols = Xs.columns

        for param, covs in cols.groupby(cols.get_level_values(0)).items():
            initial_point[param] = np.zeros(covs.shape)
            if constant_col in covs:
                initial_point[param][covs.tolist().index(constant_col)] = _transform_ith_param(getattr(uni_model, param))
        return initial_point

    def plot(self, columns=None, parameter=None, ax=None, **errorbar_kwargs):
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

        z = utils.inv_normal_cdf(1 - self.alpha / 2)

        params_ = self.params_.copy()
        standard_errors_ = self.standard_errors_.copy()
        user_supplied_columns = False

        if columns is not None:
            params_ = params_.loc[:, columns]
            standard_errors_ = standard_errors_.loc[:, columns]
            user_supplied_columns = True
        if parameter is not None:
            params_ = params_.loc[parameter]
            standard_errors_ = standard_errors_.loc[parameter]

        columns = params_.index

        hazards = params_.loc[columns].to_frame(name="coefs")
        hazards["se"] = z * standard_errors_.loc[columns]

        if not user_supplied_columns:
            if isinstance(hazards.index, pd.MultiIndex):
                hazards = hazards.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by="coefs", ascending=True))
            else:
                hazards = hazards.sort_values(by="coefs", ascending=True)

        yaxis_locations = list(range(len(columns)))

        ax.errorbar(hazards["coefs"], yaxis_locations, xerr=hazards["se"], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
        ax.set_ylim(best_ylim)

        if isinstance(columns[0], tuple):
            tick_labels = ["%s: %s" % (c, p) for (p, c) in hazards.index]
        else:
            tick_labels = [i for i in hazards.index]

        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("log(accelerated failure rate) (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def plot_partial_effects_on_outcome(
        self, covariates, values, plot_baseline=True, times=None, y="survival_function", ax=None, **kwargs
    ):
        """
        Produces a visual representation comparing the baseline survival curve of the model versus
        what happens when a covariate(s) is varied over values in a group. This is useful to compare
        subjects' survival as we vary covariate(s), all else being held equal. The baseline survival
        curve is equal to the predicted survival curve at all average values in the original dataset.

        Parameters
        ----------
        covariates: string or list
            a string (or list of strings) of the covariate in the original dataset that we wish to vary.
        values: 1d or 2d iterable
            an iterable of the values we wish the covariate to take on.
        plot_baseline: bool
            also display the baseline survival, defined as the survival at the mean of the original dataset.
        times: iterable
            pass in a times to plot
        y: str
            one of "survival_function", "hazard", "cumulative_hazard". Default "survival_function"
        kwargs:
            pass in additional plotting commands

        Returns
        -------
        ax: matplotlib axis, or list of axis'
            the matplotlib axis that be edited.

        Examples
        ---------

        .. code:: python

            from lifelines import datasets, WeibullAFTFitter
            rossi = datasets.load_rossi()
            wf = WeibullAFTFitter().fit(rossi, 'week', 'arrest')
            wf.plot_partial_effects_on_outcome('prio', values=np.arange(0, 15), cmap='coolwarm')

            # multiple variables at once
            wf.plot_partial_effects_on_outcome(['prio', 'paro'], values=[[0, 0], [5, 0], [10, 0], [0, 1], [5, 1], [10, 1]], cmap='coolwarm', y="hazard")

        """
        from matplotlib import pyplot as plt

        covariates = utils._to_list(covariates)
        values = np.atleast_1d(values)
        if len(values.shape) == 1:
            values = values[None, :].T

        if len(covariates) != values.shape[1]:
            raise ValueError("The number of covariates must equal to second dimension of the values array.")

        original_columns = self._central_values.columns
        for covariate in covariates:
            if covariate not in original_columns:
                raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        if ax is None:
            ax = plt.gca()

        # model X
        x_bar = self._central_values
        X = pd.concat([x_bar] * values.shape[0])
        if np.array_equal(np.eye(len(covariates)), values):
            X.index = ["%s=1" % c for c in covariates]
        else:
            X.index = [", ".join("%s=%s" % (c, v) for (c, v) in zip(covariates, row)) for row in values]

        for covariate, value in zip(covariates, values.T):
            X[covariate] = value

        # model ancillary X
        x_bar_anc = self._central_values
        ancillary_X = pd.concat([x_bar_anc] * values.shape[0])
        for covariate, value in zip(covariates, values.T):
            ancillary_X[covariate] = value

        # if a column is typeA in the dataset, but the user gives us typeB, we want to cast it. This is
        # most relevant for categoricals.
        X = X.astype(self._central_values.dtypes)
        ancillary_X = ancillary_X.astype(self._central_values.dtypes)

        getattr(self, "predict_%s" % y)(X, ancillary=ancillary_X, times=times).plot(ax=ax, **kwargs)
        if plot_baseline:
            getattr(self, "predict_%s" % y)(x_bar, ancillary=x_bar_anc, times=times).rename(columns={0: "baseline %s" % y}).plot(
                ax=ax, ls=":", color="k"
            )
        return ax

    def _prep_inputs_for_prediction_and_return_scores(self, X, ancillary_X):
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            Xs = self.regressors.transform_df(X)
            primary_X = Xs[self._primary_parameter_name]
            ancillary_X = Xs[self._ancillary_parameter_name]
        elif isinstance(X, pd.Series):
            return self._prep_inputs_for_prediction_and_return_scores(X.to_frame().T.infer_objects(), ancillary_X)
        else:
            assert X.shape[1] == self.params_.loc[self._primary_parameter_name].shape[0]

        primary_params = self.params_[self._primary_parameter_name]
        ancillary_params = self.params_[self._ancillary_parameter_name]

        primary_scores = np.exp(primary_X.astype(float) @ primary_params)
        ancillary_scores = np.exp(ancillary_X.astype(float) @ ancillary_params)

        return primary_scores, ancillary_scores

    def predict_survival_function(self, df, times=None, conditional_after=None, ancillary=None) -> pd.DataFrame:
        """
        Predict the survival function for individuals, given their covariates. This assumes that the individual
        just entered the study (that is, we do not condition on how long they have already lived for.)

        Parameters
        ----------

        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary:
            supply an dataframe to regress ancillary parameters against, if necessary.
        times: iterable, optional
            an iterable of increasing times to predict the survival function at. Default
            is the set of all durations (observed and unobserved).
        conditional_after: iterable, optional
            Must be equal is size to df.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.
        """
        with np.errstate(divide="ignore"):
            return np.exp(
                -self.predict_cumulative_hazard(df, ancillary=ancillary, times=times, conditional_after=conditional_after)
            )

    def predict_median(self, df, *, ancillary=None, conditional_after=None) -> pd.DataFrame:
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

        Parameters
        ----------
        df: DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary:
            supply an dataframe to regress ancillary parameters against, if necessary.
        conditional_after: iterable, optional
            Must be equal is size to df.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.


        See Also
        --------
        predict_percentile, predict_expectation

        """

        return self.predict_percentile(df, ancillary=ancillary, p=0.5, conditional_after=conditional_after)

    def predict_percentile(self, df, *, ancillary=None, p=0.5, conditional_after=None) -> pd.Series:
        warnings.warn(
            "Approximating using `predict_survival_function`. To increase accuracy, try using or increasing the resolution of the timeline kwarg in `.fit(..., timeline=timeline)`.\n",
            exceptions.ApproximationWarning,
        )
        return utils.qth_survival_times(
            p, self.predict_survival_function(df, ancillary=ancillary, conditional_after=conditional_after)
        )

    def predict_hazard(self, df, *, ancillary=None, times=None, conditional_after=None) -> pd.DataFrame:
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

        Parameters
        ----------
        df: DataFrame
            a (n,d) covariate numpy array, Series, or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary:
            supply an dataframe to regress ancillary parameters against, if necessary.
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved).
        conditional_after: iterable, optional
            Not implemented yet

        See Also
        --------
        predict_percentile, predict_expectation, predict_survival_function
        """

        if isinstance(df, pd.Series):
            df = df.to_frame().T.infer_objects()

        df = df.copy()
        times = utils.coalesce(times, self.timeline)
        times = np.atleast_1d(times).astype(float)

        if isinstance(df, pd.Series):
            df = df.to_frame().T.infer_objects()

        n = df.shape[0]

        if isinstance(ancillary, pd.DataFrame):
            assert ancillary.shape[0] == df.shape[0], "ancillary must be the same shape[0] as df"
            for c in ancillary.columns.difference(df.columns):
                df[c] = ancillary[c]

        Xs = utils.DataframeSlicer(self.regressors.transform_df(df))

        params_dict = {parameter_name: self.params_.loc[parameter_name].values for parameter_name in self._fitted_parameter_names}

        if conditional_after is None:
            return pd.DataFrame(self._hazard(params_dict, np.tile(times, (n, 1)).T, Xs), index=times, columns=df.index)
        else:
            conditional_after = np.asarray(conditional_after)
            times_to_evaluate_at = (conditional_after.reshape((n, 1)) + np.tile(times, (n, 1))).T
            return pd.DataFrame(self._hazard(params_dict, times_to_evaluate_at, Xs), index=times, columns=df.index)

    def predict_cumulative_hazard(self, df, *, ancillary=None, times=None, conditional_after=None) -> pd.DataFrame:
        """
        Predict the cumulative hazard for the individuals.

        Parameters
        ----------
        df: DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.
        ancillary:
            supply an dataframe to regress ancillary parameters against, if necessary.
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved).
        conditional_after: iterable, optional
            Must be equal is size to df.shape[0] (denoted `n` above).  An iterable (array, list, series) of possibly non-zero values that represent how long the
            subject has already lived for. Ex: if :math:`T` is the unknown event time, then this represents
            :math:`T | T > s`. This is useful for knowing the *remaining* hazard/survival of censored subjects.
            The new timeline is the remaining duration of the subject, i.e. normalized back to starting at 0.


        See Also
        --------
        predict_percentile, predict_expectation, predict_survival_function
        """
        if isinstance(df, pd.Series):
            df = df.to_frame().T.infer_objects()

        df = df.copy()
        times = utils.coalesce(times, self.timeline)
        times = np.atleast_1d(times).astype(float)

        n = df.shape[0]

        if isinstance(ancillary, pd.DataFrame):
            assert ancillary.shape[0] == df.shape[0], "ancillary must be the same shape[0] as df"
            for c in ancillary.columns.difference(df.columns):
                df[c] = ancillary[c]

        Xs = utils.DataframeSlicer(self.regressors.transform_df(df))

        params_dict = {parameter_name: self.params_.loc[parameter_name].values for parameter_name in self._fitted_parameter_names}

        if conditional_after is None:
            times_to_evaluate_at = np.tile(times, (n, 1)).T
            return pd.DataFrame(self._cumulative_hazard(params_dict, times_to_evaluate_at, Xs), index=times, columns=df.index)
        else:
            conditional_after = np.asarray(conditional_after)
            times_to_evaluate_at = (conditional_after.reshape((n, 1)) + np.tile(times, (n, 1))).T
            return pd.DataFrame(
                np.clip(
                    self._cumulative_hazard(params_dict, times_to_evaluate_at, Xs)
                    - self._cumulative_hazard(params_dict, conditional_after, Xs),
                    0,
                    np.inf,
                ),
                index=times,
                columns=df.index,
            )

    def compute_residuals(self, training_dataframe: pd.DataFrame, kind: str) -> pd.DataFrame:
        raise NotImplementedError("Working on it. Only available for Cox models at the moment.")
