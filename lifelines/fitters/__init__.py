# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import collections
from functools import wraps
import sys
import warnings

# pylint: disable=wrong-import-position
warnings.simplefilter(action="ignore", category=FutureWarning)

from textwrap import dedent

import numpy as np
import autograd.numpy as anp
from autograd import hessian, value_and_grad, elementwise_grad as egrad
from autograd.differential_operators import make_jvp_reversemode
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from numpy.linalg import inv, pinv


from lifelines.plotting import _plot_estimate
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
    StatisticalWarning,
    StatError,
    median_survival_times,
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

    def _negative_log_likelihood(self, params, T, E, entry):
        n = T.shape[0]
        hz = self._hazard(params, T[E])
        hz = anp.clip(hz, 1e-18, np.inf)

        ll = (
            (anp.log(hz)).sum()
            - self._cumulative_hazard(params, T).sum()
            + self._cumulative_hazard(params, entry).sum()
        )
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
    def plot_cumulative_hazard(self, **kwargs):
        return self.plot(**kwargs)

    @_must_call_fit_first
    def plot_survival_function(self, **kwargs):
        return _plot_estimate(
            self,
            estimate=getattr(self, "survival_function_"),
            confidence_intervals=self.confidence_interval_survival_function_,
            **kwargs
        )

    @_must_call_fit_first
    def plot_hazard(self, **kwargs):
        return _plot_estimate(
            self, estimate=getattr(self, "hazard_"), confidence_intervals=self.confidence_interval_hazard_, **kwargs
        )


class KnownModelParametericUnivariateFitter(ParametericUnivariateFitter):

    _KNOWN_MODEL = True


class ParametericRegressionFitter(BaseFitter):
    pass
