# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import collections
from functools import wraps
import sys
import warnings
from datetime import datetime

# pylint: disable=wrong-import-position
warnings.simplefilter(action="ignore", category=FutureWarning)

from textwrap import dedent

import numpy as np
import autograd.numpy as anp
from autograd import hessian, value_and_grad, elementwise_grad as egrad, grad
from autograd.differential_operators import make_jvp_reversemode
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from numpy.linalg import inv, pinv


from lifelines.plotting import _plot_estimate, set_kwargs_drawstyle, set_kwargs_ax

from lifelines.utils import (
    qth_survival_times,
    _to_array,
    dataframe_interpolate_at_times,
    ConvergenceError,
    inv_normal_cdf,
    string_justify,
    format_floats,
    format_p_value,
    coalesce,
    check_nans_or_infs,
    pass_for_numeric_dtypes_or_raise_array,
    check_for_numeric_dtypes_or_raise,
    check_complete_separation,
    check_low_var,
    StatisticalWarning,
    StatError,
    median_survival_times,
    normalize,
    concordance_index,
)
from lifelines.compat import PY2, PY3


__all__ = []


def _must_call_fit_first(func):
    @wraps(func)
    def error_wrapper(*args, **kwargs):
        self = args[0]
        try:
            self._predict_label
        except AttributeError:
            raise RuntimeError("Must call `fit` first!")
        return func(*args, **kwargs)

    return error_wrapper


class BaseFitter(object):
    def __init__(self, alpha=0.05):
        if not (0 < alpha <= 1.0):
            raise ValueError("alpha parameter must be between 0 and 1.")
        self.alpha = alpha

    def __repr__(self):
        classname = self.__class__.__name__
        try:
            s = """<lifelines.%s: fitted with %d observations, %d censored>""" % (
                classname,
                self.event_observed.shape[0],
                self.event_observed.shape[0] - np.where(self.event_observed)[0].shape[0],
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s


class UnivariateFitter(BaseFitter):
    @_must_call_fit_first
    def _update_docstrings(self):
        # Update their docstrings
        if PY2:
            self.__class__.subtract.__func__.__doc__ = self.subtract.__doc__.format(
                self._estimate_name, self.__class__.__name__
            )
            self.__class__.divide.__func__.__doc__ = self.divide.__doc__.format(
                self._estimate_name, self.__class__.__name__
            )
            self.__class__.predict.__func__.__doc__ = self.predict.__doc__.format(self.__class__.__name__)
            self.__class__.plot.__func__.__doc__ = _plot_estimate.__doc__.format(
                self.__class__.__name__, self._estimate_name
            )
        elif PY3:
            self.__class__.subtract.__doc__ = self.subtract.__doc__.format(self._estimate_name, self.__class__.__name__)
            self.__class__.divide.__doc__ = self.divide.__doc__.format(self._estimate_name, self.__class__.__name__)
            self.__class__.predict.__doc__ = self.predict.__doc__.format(self.__class__.__name__)
            self.__class__.plot.__doc__ = _plot_estimate.__doc__.format(self.__class__.__name__, self._estimate_name)

    @_must_call_fit_first
    def plot(self, **kwargs):
        return _plot_estimate(
            self, estimate=getattr(self, self._estimate_name), confidence_intervals=self.confidence_interval_, **kwargs
        )

    @_must_call_fit_first
    def subtract(self, other):
        """
        Subtract the {0} of two {1} objects.

        Parameters
        ----------
        other: an {1} fitted instance.
        """
        self_estimate = getattr(self, self._estimate_name)
        other_estimate = getattr(other, other._estimate_name)
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        return pd.DataFrame(
            self_estimate.reindex(new_index, method="ffill").values
            - other_estimate.reindex(new_index, method="ffill").values,
            index=new_index,
            columns=["diff"],
        )

    @_must_call_fit_first
    def divide(self, other):
        """
        Divide the {0} of two {1} objects.

        Parameters
        ----------
        other: an {1} fitted instance.

        """
        self_estimate = getattr(self, self._estimate_name)
        other_estimate = getattr(other, other._estimate_name)
        new_index = np.concatenate((other_estimate.index, self_estimate.index))
        new_index = np.unique(new_index)
        t = pd.DataFrame(
            self_estimate.reindex(new_index, method="ffill").values
            / other_estimate.reindex(new_index, method="ffill").values,
            index=new_index,
            columns=["ratio"],
        )
        return t

    @_must_call_fit_first
    def predict(self, times):
        """
        Predict the {0} at certain point in time. Uses a linear interpolation if
        points in time are not in the index.

        Parameters
        ----------
        times: a scalar or an array of times to predict the value of {0} at.

        Returns
        -------
        predictions: a scalar if time is a scalar, a numpy array if time in an array.
        """
        if callable(self._estimation_method):
            return pd.DataFrame(self._estimation_method(_to_array(times)), index=_to_array(times)).loc[times].squeeze()
        estimate = getattr(self, self._estimation_method)
        # non-linear interpolations can push the survival curves above 1 and below 0.
        return dataframe_interpolate_at_times(estimate, times)

    @property
    @_must_call_fit_first
    def conditional_time_to_event_(self):
        """
        Return a DataFrame, with index equal to ``survival_function_``, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        individual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.

        Returns
        -------
        conditional_time_to_: DataFrame
            with index equal to ``survival_function_``

        """
        return self._conditional_time_to_event_()

    @_must_call_fit_first
    def _conditional_time_to_event_(self):
        """
        Return a DataFrame, with index equal to survival_function_, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        individual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.

        Returns
        -------
        conditional_time_to_: DataFrame
            with index equal to survival_function_

        """
        age = self.survival_function_.index.values[:, None]
        columns = ["%s - Conditional time remaining to event" % self._label]
        return (
            pd.DataFrame(
                qth_survival_times(self.survival_function_[self._label] * 0.5, self.survival_function_)
                .sort_index(ascending=False)
                .values,
                index=self.survival_function_.index,
                columns=columns,
            )
            - age
        )

    @_must_call_fit_first
    def hazard_at_times(self, times, label=None):
        raise NotImplementedError

    @_must_call_fit_first
    def survival_function_at_times(self, times, label=None):
        raise NotImplementedError

    @_must_call_fit_first
    def cumulative_hazard_at_times(self, times, label=None):
        raise NotImplementedError

    @_must_call_fit_first
    def plot_cumulative_hazard(self, **kwargs):
        raise NotImplementedError()

    @_must_call_fit_first
    def plot_survival_function(self, **kwargs):
        raise NotImplementedError()

    @_must_call_fit_first
    def plot_hazard(self, **kwargs):
        raise NotImplementedError()


class ParametericUnivariateFitter(UnivariateFitter):
    """
    Without overriding anything, assumes all parameters must be greater than 0.

    """

    _KNOWN_MODEL = False
    _MIN_PARAMETER_VALUE = 1e-09

    def __init__(self, *args, **kwargs):
        super(ParametericUnivariateFitter, self).__init__(*args, **kwargs)
        self._estimate_name = "cumulative_hazard_"
        if not hasattr(self, "_hazard"):
            # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            self._hazard = egrad(self._cumulative_hazard, argnum=1)
        if not hasattr(self, "_bounds"):
            self._bounds = [(0.0, None)] * len(self._fitted_parameter_names)
        self._bounds = list(self._buffer_bounds(self._bounds))

        if not hasattr(self, "_initial_values"):
            self._initial_values = np.array(list(self._initial_values_from_bounds()))

        if "alpha" in self._fitted_parameter_names:
            raise NameError("'alpha' in _fitted_parameter_names is a lifelines reserved word. Try 'alpha_' instead.")

        if len(self._bounds) != len(self._fitted_parameter_names) != self._initial_values.shape[0]:
            raise ValueError(
                "_bounds must be the same shape as _fitted_parameters_names must be the same shape as _initial_values"
            )

    def _check_cumulative_hazard_is_monotone_and_positive(self, durations, values):
        class_name = self.__class__.__name__

        cumulative_hazard = self._cumulative_hazard(values, durations)
        if not np.all(cumulative_hazard > 0):
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
                StatisticalWarning,
            )

        derivative_of_cumulative_hazard = self._hazard(values, durations)
        if not np.all(derivative_of_cumulative_hazard >= 0):
            warnings.warn(
                dedent(
                    """\
                Cumulative hazard is not strictly non-decreasing. For example, try:

                >>> fitter = {0}()
                >>> fitter._hazard({1}, np.sort(durations))

                This may harm convergence, or return nonsensical results.
            """.format(
                        class_name, values.__repr__()
                    )
                ),
                StatisticalWarning,
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

    def _buffer_bounds(self, bounds):
        for (lb, ub) in bounds:
            if lb is None and ub is None:
                yield (None, None)
            elif lb is None:
                yield (None, ub - self._MIN_PARAMETER_VALUE)
            elif ub is None:
                yield (lb + self._MIN_PARAMETER_VALUE, None)
            else:
                yield (lb + self._MIN_PARAMETER_VALUE, ub - self._MIN_PARAMETER_VALUE)

    def _cumulative_hazard(self, params, times):
        raise NotImplementedError

    def _survival_function(self, params, times):
        return anp.exp(-self._cumulative_hazard(params, times))

    def _log_hazard(self, params, times):
        hz = self._hazard(params, times)
        hz = anp.clip(hz, 1e-18, np.inf)
        return anp.log(hz)

    def _negative_log_likelihood(self, params, T, E, entry):
        n = T.shape[0]
        log_hz = self._log_hazard(params, T[E])

        ll = log_hz.sum() - self._cumulative_hazard(params, T).sum() + self._cumulative_hazard(params, entry).sum()
        return -ll / n

    def _compute_confidence_bounds_of_cumulative_hazard(self, alpha, ci_labels):
        return self._compute_confidence_bounds_of_transform(self._cumulative_hazard, alpha, ci_labels)

    def _compute_confidence_bounds_of_transform(self, transform, alpha, ci_labels):
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

        """
        alpha2 = 1 - alpha / 2.0
        z = inv_normal_cdf(alpha2)
        df = pd.DataFrame(index=self.timeline)

        # pylint: disable=no-value-for-parameter
        gradient_of_cum_hazard_at_mle = make_jvp_reversemode(transform)(
            self._fitted_parameters_, self.timeline.astype(float)
        )

        gradient_at_times = np.vstack(
            [gradient_of_cum_hazard_at_mle(basis) for basis in np.eye(len(self._fitted_parameters_), dtype=float)]
        )

        std_cumulative_hazard = np.sqrt(
            np.einsum("nj,jk,nk->n", gradient_at_times.T, self.variance_matrix_, gradient_at_times.T)
        )

        if ci_labels is None:
            ci_labels = ["%s_upper_%g" % (self._label, 1 - alpha), "%s_lower_%g" % (self._label, 1 - alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = transform(self._fitted_parameters_, self.timeline) + z * std_cumulative_hazard
        df[ci_labels[1]] = transform(self._fitted_parameters_, self.timeline) - z * std_cumulative_hazard
        return df

    def _fit_model(self, T, E, entry, show_progress=True):

        non_zero_entries = entry[entry > 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = minimize(
                value_and_grad(self._negative_log_likelihood),  # pylint: disable=no-value-for-parameter
                self._initial_values,
                jac=True,
                method="L-BFGS-B",
                args=(T, E, non_zero_entries),
                bounds=self._bounds,
                options={"disp": show_progress},
            )

            if results.success:
                # pylint: disable=no-value-for-parameter
                hessian_ = hessian(self._negative_log_likelihood)(results.x, T, E, non_zero_entries)
                return results.x, -results.fun * T.shape[0], T.shape[0] * hessian_
            print(results)
            if self._KNOWN_MODEL:
                raise ConvergenceError(
                    dedent(
                        """\
                    Fitting did not converge. This is mostly a lifelines problem, but a few things you can check:
                    1. Are there any extreme values in the durations column?
                      - Try scaling your durations to a more reasonable values closer to 1 (multipling or dividing by some 10^n).
                      - Try dropping them to see if the model converges.
                """
                    )
                )

            else:
                raise ConvergenceError(
                    dedent(
                        """\
                    Fitting did not converge.

                    1. Are two parameters in the model colinear / exchangeable? (Change model)
                    2. Is the cumulative hazard always non-negative and always non-decreasing? (Assumption error)
                    3. Are there inputs to the cumulative hazard that could produce nans or infs? (Check your _bounds)

                    This could be a problem with your data:
                    1. Are there any extreme values in the durations column?
                        - Try scaling your durations to a more reasonable value closer to 1 (multipling or dividing by a large constant).
                        - Try dropping them to see if the model converges.
                    """
                    )
                )

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _estimation_method(self, t):
        return self.survival_function_at_times(t)

    def _compute_standard_errors(self):
        return pd.DataFrame(
            [np.sqrt(self.variance_matrix_.diagonal())], index=["se"], columns=self._fitted_parameter_names
        )

    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().loc["se"]
        z = inv_normal_cdf(1 - self.alpha / 2.0)
        return pd.DataFrame(
            [self._fitted_parameters_ + z * se, self._fitted_parameters_ - z * se],
            columns=self._fitted_parameter_names,
            index=["upper-bound", "lower-bound"],
        )

    def _compute_z_values(self):
        return (self._fitted_parameters_ - self._initial_values) / self._compute_standard_errors().loc["se"]

    @property
    @_must_call_fit_first
    def summary(self):
        """
        Summary statistics describing the fit.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper

        See Also
        --------
        ``print_summary``
        """
        ci = 1 - self.alpha
        lower_upper_bounds = self._compute_confidence_bounds_of_parameters()
        df = pd.DataFrame(index=self._fitted_parameter_names)
        df["coef"] = self._fitted_parameters_
        df["se(coef)"] = self._compute_standard_errors().loc["se"]
        df["lower %g" % ci] = lower_upper_bounds.loc["lower-bound"]
        df["upper %g" % ci] = lower_upper_bounds.loc["upper-bound"]
        df["p"] = self._compute_p_values()
        with np.errstate(invalid="ignore", divide="ignore"):
            df["-log2(p)"] = -np.log2(df["p"])
        return df

    @_must_call_fit_first
    def print_summary(self, decimals=2, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """
        justify = string_justify(18)
        print(self)
        print("{} = {}".format(justify("number of subjects"), self.durations.shape[0]))
        print("{} = {}".format(justify("number of events"), np.where(self.event_observed)[0].shape[0]))
        print("{} = {:.3f}".format(justify("log-likelihood"), self._log_likelihood))
        print(
            "{} = {}".format(
                justify("hypothesis"),
                ", ".join(
                    "%s != %d" % (name, iv) for (name, iv) in zip(self._fitted_parameter_names, self._initial_values)
                ),
            )
        )

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))

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
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
        durations: an array, or pd.Series
          length n, duration subject was observed for
        event_observed: numpy array or pd.Series, optional
          length n, True if the the death was observed, False if the event was lost (right-censored). Defaults all True if event_observed==None
        timeline: list, optional
            return the estimate at the values in timeline (postively increasing)
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: list, optional
            add custom column names to the generated confidence intervals as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
        show_progress: boolean, optional
            since this is an iterative fitting algorithm, switching this to True will display some iteration details.
        entry: an array, or pd.Series, of length n
            relative time when a subject entered the study. This is useful for left-truncated (not left-censored) observations. If None, all members of the population
            entered study when they were "born": time zero.

        Returns
        -------
          self
            self with new properties like ``cumulative_hazard_``, ``survival_function_``

        """
        label = coalesce(label, self.__class__.__name__.replace("Fitter", "") + "_estimate")

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(pass_for_numeric_dtypes_or_raise_array(durations))
        # check for negative or 0 durations - these are not allowed in a weibull model.
        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        if not self._KNOWN_MODEL:
            self._check_cumulative_hazard_is_monotone_and_positive(self.durations, self._initial_values)

        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )

        self.entry = np.asarray(entry) if entry is not None else np.zeros_like(self.durations)

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline).astype(float))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label
        self._ci_labels = ci_labels
        self.alpha = coalesce(alpha, self.alpha)

        # estimation
        self._fitted_parameters_, self._log_likelihood, self._hessian_ = self._fit_model(
            self.durations, self.event_observed.astype(bool), self.entry, show_progress=show_progress
        )

        if not self._KNOWN_MODEL:
            self._check_cumulative_hazard_is_monotone_and_positive(self.durations, self._fitted_parameters_)

        for param_name, fitted_value in zip(self._fitted_parameter_names, self._fitted_parameters_):
            setattr(self, param_name, fitted_value)

        try:
            self.variance_matrix_ = inv(self._hessian_)
        except np.linalg.LinAlgError:
            self.variance_matrix_ = pinv(self._hessian_)
            warning_text = dedent(
                """\

                The hessian was not invertable. This could be a model problem:

                1. Are two parameters in the model colinear / exchangeable?
                2. Is the cumulative hazard always non-negative and always non-decreasing?
                3. Are there cusps/ in the cumulative hazard?

                We will instead approximate it using the psuedo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the
                fitted parameters too. Perform plots of the cumulative hazard to help understand
                the latter's bias.
                """
            )
            warnings.warn(warning_text, StatisticalWarning)

        self._predict_label = label
        self._update_docstrings()

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame()
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame()
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame()

        return self

    @_must_call_fit_first
    def survival_function_at_times(self, times, label=None):
        """
        Return a Pandas series of the predicted survival value at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the survival function at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        Returns
        --------
        pd.Series

        """
        label = coalesce(label, self._label)
        return pd.Series(self._survival_function(self._fitted_parameters_, times), index=_to_array(times), name=label)

    @_must_call_fit_first
    def cumulative_hazard_at_times(self, times, label=None):
        """
        Return a Pandas series of the predicted cumulative hazard value at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the cumulative hazard at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        Returns
        --------
        pd.Series

        """
        label = coalesce(label, self._label)
        return pd.Series(self._cumulative_hazard(self._fitted_parameters_, times), index=_to_array(times), name=label)

    @_must_call_fit_first
    def hazard_at_times(self, times, label=None):
        """
        Return a Pandas series of the predicted hazard at specific times.

        Parameters
        -----------
        times: iterable or float
          values to return the hazard at.
        label: string, optional
          Rename the series returned. Useful for plotting.

        Returns
        --------
        pd.Series

        """
        label = coalesce(label, self._label)
        return pd.Series(self._hazard(self._fitted_parameters_, times), index=_to_array(times), name=label)

    @property
    @_must_call_fit_first
    def median_(self):
        """
        Return the unique time point, t, such that S(t) = 0.5. This is the "half-life" of the population, and a
        robust summary statistic for the population, if it exists.
        """
        return median_survival_times(self.survival_function_)

    @property
    @_must_call_fit_first
    def confidence_interval_(self):
        """
        The confidence interval of the cumulative hazard. This is an alias for ``confidence_interval_cumulative_hazard_``.
        """
        return self._compute_confidence_bounds_of_cumulative_hazard(self.alpha, self._ci_labels)

    @property
    @_must_call_fit_first
    def confidence_interval_cumulative_hazard_(self):
        """
        The confidence interval of the cumulative hazard. This is an alias for ``confidence_interval_``.
        """
        return self.confidence_interval_

    @property
    @_must_call_fit_first
    def confidence_interval_hazard_(self):
        """
        The confidence interval of the hazard.
        """
        return self._compute_confidence_bounds_of_transform(self._hazard, self.alpha, self._ci_labels)

    @property
    @_must_call_fit_first
    def confidence_interval_survival_function_(self):
        """
        The confidence interval of the survival function.
        """
        return self._compute_confidence_bounds_of_transform(self._survival_function, self.alpha, self._ci_labels)

    @_must_call_fit_first
    def plot(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(
            self, estimate=getattr(self, self._estimate_name), confidence_intervals=self.confidence_interval_, **kwargs
        )

    @_must_call_fit_first
    def plot_cumulative_hazard(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return self.plot(**kwargs)

    @_must_call_fit_first
    def plot_survival_function(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(
            self,
            estimate=getattr(self, "survival_function_"),
            confidence_intervals=self.confidence_interval_survival_function_,
            **kwargs
        )

    @_must_call_fit_first
    def plot_hazard(self, **kwargs):
        set_kwargs_drawstyle(kwargs, "default")
        return _plot_estimate(
            self, estimate=getattr(self, "hazard_"), confidence_intervals=self.confidence_interval_hazard_, **kwargs
        )


class KnownModelParametericUnivariateFitter(ParametericUnivariateFitter):

    _KNOWN_MODEL = True


class ParametericRegressionFitter(BaseFitter):
    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio=0.0, fit_intercept=True):
        super(ParametericRegressionFitter, self).__init__(alpha=alpha)
        self._hazard = egrad(self._cumulative_hazard, argnum=1)  # pylint: disable=unexpected-keyword-arg
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self._fitted_parameter_names = [self._primary_parameter_name, self._ancillary_parameter_name]

    def _negative_log_likelihood(self, params, T, E, W, *Xs):
        hz = self._hazard(params, T, *Xs)
        hz = anp.clip(hz, 1e-18, np.inf)

        ll = (W * E * anp.log(hz)).sum() - (W * self._cumulative_hazard(params, T, *Xs)).sum()
        if self.penalizer > 0:
            penalty = self.l1_ratio * anp.abs(params).sum() + 0.5 * (1.0 - self.l1_ratio) * (params ** 2).sum()
        else:
            penalty = 0

        ll = ll / np.sum(W)
        return -ll + self.penalizer * penalty

    def fit(
        self,
        df,
        duration_col=None,
        event_col=None,
        ancillary_df=None,
        show_progress=False,
        timeline=None,
        weights_col=None,
        robust=False,
    ):
        """
        Fit the accelerated failure time model to a dataset.

        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).

        duration_col: string
            the name of the column in dataframe that contains the subjects'
            lifetimes.

        event_col: string, optional
            the  name of thecolumn in dataframe that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.

        show_progress: boolean, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.

        ancillary_df: None, boolean, or DataFrame, optional (default=None)
            Choose to model the ancillary parameters.
            If None or False, explicity do not fit the ancillary parameters using any covariates.
            If True, model the ancillary parameters with the same covariates as ``df``.
            If DataFrame, provide covariates to model the ancillary parameters. Must be the same row count as ``df``.

        timeline: array, optional
            Specify a timeline that will be used for plotting and prediction

        weights_col: string
            the column in df that specifies weights per observation.

        robust: boolean, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator.

        Returns
        -------
        self:
            self with additional new properties: ``print_summary``, ``params_``, ``confidence_intervals_`` and more


        Examples
        --------
        >>> from lifelines import WeibullAFTFitter
        >>>
        >>> df = pd.DataFrame({
        >>>     'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        >>>     'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        >>>     'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>> })
        >>>
        >>> aft = WeibullAFTFitter()
        >>> aft.fit(df, 'T', 'E')
        >>> aft.print_summary()
        >>> aft.predict_median(df)
        >>>
        >>> aft = WeibullAFTFitter()
        >>> aft.fit(df, 'T', 'E', ancillary_df=df)
        >>> aft.print_summary()
        >>> aft.predict_median(df)

        """
        if duration_col is None:
            raise TypeError("duration_col cannot be None.")

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self.weights_col = weights_col
        self._n_examples = df.shape[0]
        self.timeline = timeline
        self.robust = robust

        df = df.copy()

        T = pass_for_numeric_dtypes_or_raise_array(df.pop(duration_col)).astype(float)
        E = (
            pass_for_numeric_dtypes_or_raise_array(df.pop(self.event_col)).astype(bool)
            if (self.event_col is not None)
            else pd.Series(np.ones(self._n_examples, dtype=bool), index=df.index, name="E")
        )
        weights = (
            pass_for_numeric_dtypes_or_raise_array(df.pop(self.weights_col)).astype(float)
            if (self.weights_col is not None)
            else pd.Series(np.ones(self._n_examples, dtype=float), index=df.index, name="weights")
        )
        # check to make sure their weights are okay
        if self.weights_col:
            if (weights.astype(int) != weights).any() and not self.robust:
                warnings.warn(
                    dedent(
                        """It appears your weights are not integers, possibly propensity or sampling scores then?
                                        It's important to know that the naive variance estimates of the coefficients are biased. Instead a) set `robust=True` in the call to `fit`, or b) use Monte Carlo to
                                        estimate the variances. See paper "Variance estimation when using inverse probability of treatment weighting (IPTW) with survival analysis"""
                    ),
                    StatisticalWarning,
                )
            if (weights <= 0).any():
                raise ValueError("values in weight column %s must be positive." % self.weights_col)

        self.durations = T.copy()
        self.event_observed = E.copy()
        self.weights = weights.copy()

        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        self._check_values(df, T, E, self.event_col)

        if isinstance(ancillary_df, pd.DataFrame):
            assert ancillary_df.shape[0] == df.shape[0], "ancillary_df must be the same shape[0] as df"

            ancillary_df = ancillary_df.copy().drop([duration_col, event_col], axis=1, errors="ignore")
            self._check_values(ancillary_df, T, E, self.event_col)

        elif (ancillary_df is None) or (ancillary_df is False):
            ancillary_df = pd.DataFrame(np.ones((df.shape[0],)), index=df.index, columns=["_intercept"])
        elif ancillary_df is True:
            ancillary_df = df.copy()

        if self.fit_intercept:
            assert "_intercept" not in df
            ancillary_df["_intercept"] = 1.0
            df["_intercept"] = 1.0

        self._LOOKUP_SLICE = self._create_slicer(len(df.columns), len(ancillary_df.columns))

        _norm_std, _norm_std_ancillary = df.std(0), ancillary_df.std(0)
        self._norm_mean, self._norm_mean_ancillary = df.mean(0), ancillary_df.mean(0)
        # if we included an intercept, we need to fix not divide by zero.
        if self.fit_intercept:
            _norm_std["_intercept"] = 1.0
            _norm_std_ancillary["_intercept"] = 1.0
        else:
            _norm_std[_norm_std < 1e-8] = 1.0
            _norm_std_ancillary[_norm_std_ancillary < 1e-8] = 1.0

        _index = pd.MultiIndex.from_tuples(
            [(self._primary_parameter_name, c) for c in df.columns]
            + [(self._ancillary_parameter_name, c) for c in ancillary_df.columns]
        )

        self._norm_std = pd.Series(np.append(_norm_std, _norm_std_ancillary), index=_index)

        _params, self._log_likelihood, self._hessian_ = self._fit_model(
            T.values,
            E.values,
            weights.values,
            normalize(df, 0, _norm_std).values,
            normalize(ancillary_df, 0, _norm_std_ancillary).values,
            show_progress=show_progress,
        )
        self.params_ = _params / self._norm_std

        self.variance_matrix_ = self._compute_variance_matrix()
        self.standard_errors_ = self._compute_standard_errors(
            T.values, E.values, weights.values, df.values, ancillary_df.values
        )
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self._predicted_median = self.predict_median(df, ancillary_df)

        return self

    def _check_values(self, df, T, E, event_col):
        check_for_numeric_dtypes_or_raise(df)
        check_nans_or_infs(T)
        check_nans_or_infs(E)
        check_nans_or_infs(df)
        check_complete_separation(df, E, T, event_col)

        if self.fit_intercept:
            check_low_var(df)

    def _fit_model(self, T, E, weights, *Xs, **kwargs):
        # TODO: move this to function kwarg when I remove py2
        show_progress = kwargs.pop("show_progress", False)
        n_params = sum([X.shape[1] for X in Xs])
        init_values = np.zeros((n_params,))
        sum_weights = weights.sum()

        results = minimize(
            value_and_grad(self._negative_log_likelihood),
            init_values,
            method=None if self.l1_ratio <= 0.0 else "L-BFGS-B",
            jac=True,
            args=(T, E, weights, Xs[0], Xs[1]),  # TODO: remove py2, (T, E, *Xs)
            options={"disp": show_progress},
        )
        if show_progress:
            print(results)

        if results.success:
            # pylint: disable=no-value-for-parameter
            hessian_ = hessian(self._negative_log_likelihood)(results.x, T, E, weights, *Xs)
            return results.x, -sum_weights * results.fun, sum_weights * hessian_
        print(results)
        name = self.__class__.__name__
        raise ConvergenceError(
            dedent(
                """\
            Fitting did not converge. This could be a problem with your data:
            1. Are there any extreme values? (Try modelling them or dropping them to see if it helps convergence)
            2. Trying adding a small penalizer (or changing it, if already present). Example: `%s(penalizer=0.01).fit(...)`
        """
                % name
            )
        )

    def _create_slicer(self, *sizes):

        lookup = {}
        position = 0

        for name, size in zip(self._fitted_parameter_names, sizes):
            lookup[name] = slice(position, position + size)
            position += size

        return lookup

    def _compute_variance_matrix(self):
        try:
            unit_scaled_variance_matrix_ = np.linalg.inv(self._hessian_)
        except np.linalg.LinAlgError:
            unit_scaled_variance_matrix_ = np.linalg.pinv(self._hessian_)
            warning_text = dedent(
                """\
                The hessian was not invertable. We will instead approximate it using the psuedo-inverse.

                It's advisable to not trust the variances reported, and to be suspicious of the
                fitted parameters too. Perform plots of the cumulative hazard to help understand
                the latter's bias.
                """
            )
            warnings.warn(warning_text, StatisticalWarning)

        return unit_scaled_variance_matrix_ / np.outer(self._norm_std, self._norm_std)

    def _compute_z_values(self):
        return self.params_ / self.standard_errors_

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _compute_standard_errors(self, T, E, weights, *Xs):
        if self.robust:
            se = np.sqrt(self._compute_sandwich_errors(T, E, weights, *Xs).diagonal())
        else:
            se = np.sqrt(self.variance_matrix_.diagonal())
        return pd.Series(se, name="se", index=self.params_.index)

    def _compute_sandwich_errors(self, T, E, weights, *Xs):

        ll_gradient = grad(self._negative_log_likelihood)
        params = self.params_.values
        n_params = params.shape[0]
        J = np.zeros((n_params, n_params))

        for t, e, w, x, ancillary_x in zip(T, E, weights, Xs[0], Xs[1]):
            score_vector = ll_gradient(params, t, e, w, x, ancillary_x)
            J += np.outer(score_vector, score_vector)

        return np.dot(self.variance_matrix_, J).dot(self.variance_matrix_)

    def _compute_confidence_intervals(self):
        z = inv_normal_cdf(1 - self.alpha / 2)
        se = self.standard_errors_
        params = self.params_.values
        return pd.DataFrame(
            np.c_[params - z * se, params + z * se], index=self.params_.index, columns=["lower-bound", "upper-bound"]
        )

    def _compute_likelihood_ratio_test(self):
        """
        This function computes the likelihood ratio test for the Weibull model. We
        compare the existing model (with all the covariates) to the trivial model
        of no covariates.

        """
        from lifelines.statistics import chisq_test

        ll_null = (
            self.__class__()
            .fit(pd.DataFrame({"T": self.durations, "E": self.event_observed}), "T", "E")
            ._log_likelihood
        )
        ll_alt = self._log_likelihood

        test_stat = 2 * ll_alt - 2 * ll_null
        degrees_freedom = self.params_.shape[0] - 2  # delta in number of parameters between models
        p_value = chisq_test(test_stat, degrees_freedom=degrees_freedom)
        with np.errstate(invalid="ignore", divide="ignore"):
            return test_stat, degrees_freedom, -np.log2(p_value)

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
        ci = 1 - self.alpha
        with np.errstate(invalid="ignore", divide="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["coef"] = self.params_
            df["exp(coef)"] = np.exp(self.params_)
            df["se(coef)"] = self.standard_errors_
            df["z"] = self._compute_z_values()
            df["p"] = self._compute_p_values()
            df["-log2(p)"] = -np.log2(df["p"])
            df["lower %g" % ci] = self.confidence_intervals_["lower-bound"]
            df["upper %g" % ci] = self.confidence_intervals_["upper-bound"]
            return df

    def print_summary(self, decimals=2, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        alpha: float or iterable
            specify confidence intervals to show
        kwargs:
            print additional metadata in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """

        # Print information about data first
        justify = string_justify(18)
        print(self)
        print("{} = '{}'".format(justify("duration col"), self.duration_col))
        if self.event_col:
            print("{} = '{}'".format(justify("event col"), self.event_col))
        if self.weights_col:
            print("{} = '{}'".format(justify("weights col"), self.weights_col))
        if self.penalizer > 0:
            print("{} = {}".format(justify("penalizer"), self.penalizer))
            print("{} = {}".format(justify("l1_ratio"), self.l1_ratio))

        if self.robust:
            print("{} = {}".format(justify("robust variance"), True))

        print("{} = {}".format(justify("number of subjects"), self._n_examples))
        print("{} = {}".format(justify("number of events"), self.event_observed.sum()))
        print("{} = {:.{prec}f}".format(justify("log-likelihood"), self._log_likelihood, prec=decimals))
        print("{} = {}".format(justify("time fit was run"), self._time_fit_was_called))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        # Significance codes as last column
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))

        # Significance code explanation
        print("---")
        print("Concordance = {:.{prec}f}".format(self.score_, prec=decimals))
        print(
            "Log-likelihood ratio test = {:.{prec}f} on {} df, -log2(p)={:.{prec}f}".format(
                *self._compute_likelihood_ratio_test(), prec=decimals
            )
        )

    def predict_survival_function(self, X, times=None, ancillary_X=None):
        """
        Predict the survival function for individuals, given their covariates. This assumes that the individual
        just entered the study (that is, we do not condition on how long they have already lived for.)

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
        times: iterable, optional
            an iterable of increasing times to predict the cumulative hazard at. Default
            is the set of all durations (observed and unobserved). Uses a linear interpolation if
            points in time are not in the index.


        Returns
        -------
        survival_function : DataFrame
            the survival probabilities of individuals over the timeline
        """
        return np.exp(-self.predict_cumulative_hazard(X, times=times, ancillary_X=ancillary_X))

    def predict_median(self, X, ancillary_X=None):
        """
        Predict the median lifetimes for the individuals. If the survival curve of an
        individual does not cross 0.5, then the result is infinity.

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
        predict_percentile, predict_expectation

        """
        return self.predict_percentile(X, p=0.5, ancillary_X=ancillary_X)

    @property
    def score_(self):
        """
        The concordance score (also known as the c-index) of the fit.  The c-index is a generalization of the AUC
        to survival data, including censorships.

        For this purpose, the ``score_`` is a measure of the predictive accuracy of the fitted model
        onto the training dataset.

        """
        # pylint: disable=access-member-before-definition
        if hasattr(self, "_predicted_median"):
            self._concordance_score_ = concordance_index(self.durations, self._predicted_median, self.event_observed)
            del self._predicted_median
            return self._concordance_score_
        return self._concordance_score_

    @property
    def median_survival_time_(self):
        return self.predict_median(
            self._norm_mean.to_frame().T, ancillary_X=self._norm_mean_ancillary.to_frame().T
        ).squeeze()

    @property
    def mean_survival_time_(self):
        return self.predict_expectation(
            self._norm_mean.to_frame().T, ancillary_X=self._norm_mean_ancillary.to_frame().T
        ).squeeze()

    def plot(self, columns=None, parameter=None, **errorbar_kwargs):
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

        set_kwargs_ax(errorbar_kwargs)
        ax = errorbar_kwargs.pop("ax")
        errorbar_kwargs.setdefault("c", "k")
        errorbar_kwargs.setdefault("fmt", "s")
        errorbar_kwargs.setdefault("markerfacecolor", "white")
        errorbar_kwargs.setdefault("markeredgewidth", 1.25)
        errorbar_kwargs.setdefault("elinewidth", 1.25)
        errorbar_kwargs.setdefault("capsize", 3)

        z = inv_normal_cdf(1 - self.alpha / 2)

        params_ = self.params_.copy()
        standard_errors_ = self.standard_errors_.copy()

        if columns is not None:
            params_ = params_.loc[:, columns]
            standard_errors_ = standard_errors_.loc[:, columns]
        if parameter is not None:
            params_ = params_.loc[parameter]
            standard_errors_ = standard_errors_.loc[parameter]

        columns = params_.index

        hazards = params_.loc[columns].to_frame(name="coefs")
        hazards["se"] = z * standard_errors_.loc[columns]

        if isinstance(hazards.index, pd.MultiIndex):
            hazards = hazards.groupby(level=0, group_keys=False).apply(
                lambda x: x.sort_values(by="coefs", ascending=True)
            )
        else:
            hazards = hazards.sort_values(by="coefs", ascending=True)

        yaxis_locations = list(range(len(columns)))

        ax.errorbar(hazards["coefs"], yaxis_locations, xerr=hazards["se"], **errorbar_kwargs)
        best_ylim = ax.get_ylim()
        ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
        ax.set_ylim(best_ylim)

        if isinstance(columns[0], tuple):
            tick_labels = ["%s: %s" % (c, p) for (p, c) in hazards.index]
        else:
            tick_labels = [i for i in hazards.index]

        plt.yticks(yaxis_locations, tick_labels)
        plt.xlabel("log(accelerated failure rate) (%g%% CI)" % ((1 - self.alpha) * 100))

        return ax

    def plot_covariate_groups(self, covariate, values, plot_baseline=True, **kwargs):
        """
        Produces a visual representation comparing the baseline survival curve of the model versus
        what happens when a covariate is varied over values in a group. This is useful to compare
        subjects' survival as we vary a single covariate, all else being held equal. The baseline survival
        curve is equal to the predicted survival curve at all average values in the original dataset.

        Parameters
        ----------
        covariate: string
            a string of the covariate in the original dataset that we wish to vary.
        values: iterable
            an iterable of the values we wish the covariate to take on.
        plot_baseline: bool
            also display the baseline survival, defined as the survival at the mean of the original dataset.
        kwargs:
            pass in additional plotting commands

        Returns
        -------
        ax: matplotlib axis, or list of axis'
            the matplotlib axis that be edited.
        """
        from matplotlib import pyplot as plt

        original_columns = self.params_.index.get_level_values(1)
        if covariate not in original_columns:
            raise KeyError("covariate `%s` is not present in the original dataset" % covariate)

        ax = kwargs.pop("ax", None) or plt.figure().add_subplot(111)

        x_bar = self._norm_mean.to_frame().T
        X = pd.concat([x_bar] * len(values))
        X.index = ["%s=%s" % (covariate, g) for g in values]
        X[covariate] = values

        x_bar_anc = self._norm_mean_ancillary.to_frame().T
        ancillary_X = pd.concat([x_bar_anc] * len(values))
        ancillary_X.index = ["%s=%s" % (covariate, g) for g in values]
        ancillary_X[covariate] = values

        if self.fit_intercept:
            X["_intercept"] = 1.0
            ancillary_X["_intercept"] = 1.0

        self.predict_survival_function(X, ancillary_X=ancillary_X).plot(ax=ax, **kwargs)
        if plot_baseline:
            self.predict_survival_function(x_bar, ancillary_X=x_bar_anc).rename(columns={0: "baseline survival"}).plot(
                ax=ax, ls=":", color="k"
            )
        return ax

    def _prep_inputs_for_prediction_and_return_scores(self, X, ancillary_X):
        X = X.copy()

        if ancillary_X is None:
            ancillary_X = pd.DataFrame(np.ones((X.shape[0], 1)), columns=["_intercept"])
        elif isinstance(ancillary_X, pd.DataFrame):
            ancillary_X = ancillary_X.copy()
            if self.fit_intercept:
                ancillary_X["_intercept"] = 1.0
            ancillary_X = ancillary_X[self.params_.loc[self._ancillary_parameter_name].index]
        else:
            assert ancillary_X.shape[1] == (
                self.params_.loc[self._ancillary_parameter_name].shape[0] + 1
            )  # 1 for _intercept

        if isinstance(X, pd.DataFrame):
            if self.fit_intercept:
                X["_intercept"] = 1.0
            X = X[self.params_.loc[self._primary_parameter_name].index]
        else:
            assert X.shape[1] == (self.params_.loc[self._primary_parameter_name].shape[0] + 1)  # 1 for _intercept

        primary_params = self.params_[self._LOOKUP_SLICE[self._primary_parameter_name]]
        ancillary_params = self.params_[self._LOOKUP_SLICE[self._ancillary_parameter_name]]

        primary_scores = np.exp(np.dot(X, primary_params))
        ancillary_scores = np.exp(np.dot(ancillary_X, ancillary_params))

        return primary_scores, ancillary_scores
