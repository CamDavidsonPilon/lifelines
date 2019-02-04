# -*- coding: utf-8 -*-
from __future__ import print_function
import collections
from functools import wraps
import sys
import warnings

import numpy as np
import autograd.numpy as anp
from autograd import hessian, value_and_grad, elementwise_grad as egrad
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from autograd import make_jvp


from lifelines.plotting import plot_estimate
from lifelines.utils import qth_survival_times, _to_array, dataframe_interpolate_at_times, inv_normal_cdf, string_justify, format_floats, format_p_value
from lifelines.compat import PY2, PY3


def _must_call_fit_first(func):
    @wraps(func)
    def error_wrapper(*args, **kwargs):
        self = args[0]
        try:
            self._estimate_name
        except AttributeError:
            raise RuntimeError("Must call `fit` first!")
        return func(*args, **kwargs)

    return error_wrapper


class BaseFitter(object):
    def __init__(self, alpha=0.95):
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
            self.__class__.plot.__func__.__doc__ = plot_estimate.__doc__.format(
                self.__class__.__name__, self._estimate_name
            )
        elif PY3:
            self.__class__.subtract.__doc__ = self.subtract.__doc__.format(self._estimate_name, self.__class__.__name__)
            self.__class__.divide.__doc__ = self.divide.__doc__.format(self._estimate_name, self.__class__.__name__)
            self.__class__.predict.__doc__ = self.predict.__doc__.format(self.__class__.__name__)
            self.__class__.plot.__doc__ = plot_estimate.__doc__.format(self.__class__.__name__, self._estimate_name)

    @_must_call_fit_first
    def plot(self, *args, **kwargs):
        return plot_estimate(self, *args, **kwargs)

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
        Return a DataFrame, with index equal to survival_function_, that estimates the median
        duration remaining until the death event, given survival up until time t. For example, if an
        individual exists until age 1, their expected life remaining *given they lived to time 1*
        might be 9 years.

        Returns
        -------
        conditional_time_to_: DataFrame 
            with index equal to survival_function_

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
    def hazard_at_times(self, times):
        raise NotImplementedError

    @_must_call_fit_first
    def survival_function_at_times(self, times):
        raise NotImplementedError

    @_must_call_fit_first
    def cumulative_hazard_at_times(self, times):
        raise NotImplementedError


class ParametericUnivariateFitter(UnivariateFitter):
    """
    Without overriding anything, assumes all parameters must be greater than 0.

    """
    def __init__(self, *args, **kwargs):
        super(ParametericUnivariateFitter, self).__init__(*args, **kwargs)
        self._hazard = egrad(self._cumulative_hazard, argnum=1)

    def _cumulative_hazard(self, params, times):
        raise NotImplementedError

    def _survival_function(self, params, times):
        return anp.exp(-self._cumulative_hazard(params, times))

    def _negative_log_likelihood(self, params, T, E):
        n = T.shape[0]
        ll = (E*anp.log(self._hazard(params, T))).sum() + anp.log(self._survival_function(params, T)).sum()
        return -ll / n

    def _bounds(self, alpha, ci_labels):
        alpha2 = inv_normal_cdf((1.0 + alpha) / 2.0)
        df = pd.DataFrame(index=self.timeline)

        gradient_of_cum_hazard_at_mle = make_jvp(self._cumulative_hazard)(self._fitted_parameters_, self.timeline)

        gradient_at_times = np.vstack([
            gradient_of_cum_hazard_at_mle(basis)[1] for basis in np.eye(len(self._fitted_parameters_))
        ])

        std_cumulative_hazard = np.sqrt(
            np.einsum("nj,jk,nk->n", gradient_at_times.T, self.variance_matrix_, gradient_at_times.T)
        )

        if ci_labels is None:
            ci_labels = ["%s_upper_%.2f" % (self._label, alpha), "%s_lower_%.2f" % (self._label, alpha)]
        assert len(ci_labels) == 2, "ci_labels should be a length 2 array."

        df[ci_labels[0]] = self.cumulative_hazard_at_times(self.timeline) + alpha2 * std_cumulative_hazard
        df[ci_labels[1]] = self.cumulative_hazard_at_times(self.timeline) - alpha2 * std_cumulative_hazard
        return df

    def _fit_model(self, T, E, show_progress=True):

        initial_values = np.ones(2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results = minimize(
                value_and_grad(self._negative_log_likelihood),  # pylint: disable=no-value-for-parameter
                initial_values,
                jac=True,
                method="L-BFGS-B",
                args=(T, E),
                bounds=((0.000001, None), (0.000001, None)),  # to stay well away from 0.
                options={"disp": show_progress},
            )

            if results.success:
                hessian_ = hessian(self._negative_log_likelihood)(results.x, T, E)  # pylint: disable=no-value-for-parameter
                return results.x, -results.fun, hessian_ * T.shape[0]
            print(results)
            raise ConvergenceError("Did not converge. This is a lifelines problem, not yours;")

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def survival_function_at_times(self, times):
        return pd.Series(self._survival_function(self._fitted_parameters_, times), index=_to_array(times))

    def cumulative_hazard_at_times(self, times):
        return pd.Series(self._cumulative_hazard(self._fitted_parameters_, times), index=_to_array(times))

    def hazard_at_times(self, times):
        return pd.Series(self._hazard(self._fitted_parameters_, times), index=_to_array(times))

    def _estimation_method(self, t):
        return self.survival_function_at_times(t)

    def _compute_standard_errors(self):
        return pd.DataFrame([np.sqrt(self.variance_matrix_.diagonal())], index=["se"], columns=self._fitted_parameters_names_)
    
    def _compute_confidence_bounds_of_parameters(self):
        se = self._compute_standard_errors().loc["se"]
        alpha2 = inv_normal_cdf((1.0 + self.alpha) / 2.0)
        return pd.DataFrame(
            [self._fitted_parameters_ + alpha2 * se, self._fitted_parameters_ - alpha2 * se],
            columns=self._fitted_parameters_names_,
            index=["upper-bound", "lower-bound"],
        )

    @property
    def summary(self):
        """Summary statistics describing the fit.
        Set alpha property in the object before calling.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper
        """
        lower_upper_bounds = self._compute_confidence_bounds_of_parameters()
        df = pd.DataFrame(index=self._fitted_parameters_names_)
        df["coef"] = [self.lambda_, self.rho_]
        df["se(coef)"] = self._compute_standard_errors().loc["se"]
        df["lower %.2f" % self.alpha] = lower_upper_bounds.loc["lower-bound"]
        df["upper %.2f" % self.alpha] = lower_upper_bounds.loc["upper-bound"]
        df["p"] = self._compute_p_values()
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore")
            df["-log2(p)"] = -np.log2(df["p"])
        return df

    def _compute_z_values(self):
        return (self._fitted_parameters_ - 1) / self._compute_standard_errors().loc["se"]


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
        print("{} = {}".format(justify("hypothesis"), ", ".join("%s != 1" % name for name in self._fitted_parameters_names_)))

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))