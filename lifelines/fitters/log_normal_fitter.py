# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy import log
from scipy.optimize import minimize
from scipy.integrate import cumtrapz


from lifelines.fitters import UnivariateFitter
from lifelines.utils import (
    _to_array,
    check_nans_or_infs,
    string_justify,
    format_floats,
    ConvergenceError,
    dataframe_interpolate_at_times
)

def _negative_log_likelihood(params, T, E):
    mu, sigma = params 
    Z = (log(T) - mu) / sigma
    log_sf = log(1 - norm.cdf(Z))
    
    x =  - (E * (
        norm.logpdf(Z)
        - log(T) - log(sigma) - log_sf)
    ).sum() - log_sf.sum()
    return x

class LogNormalFitter(UnivariateFitter):
    r"""
    TODO: 
    This class implements an Exponential model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = exp(-(\lambda*t)),   \lambda >0

    which implies the cumulative hazard rate is

    .. math::  H(t) = \lambda*t

    and the hazard rate is:

    .. math::  h(t) = \lambda

    After calling the `.fit` method, you have access to properties like:
     'survival_function_', 'lambda_'

    A summary of the fit is available with the method 'print_summary()'

    Notes
    -----
    Reference: https://www4.stat.ncsu.edu/~dzhang2/st745/chap3.pdf

    """

    def fit(
        self, durations, event_observed=None, timeline=None, label="LogNormal_estimate"
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
          durations: iterable
            an array, or pd.Series, of length n -- duration subject was observed for
          event_observed: iterable, optional
            an array, list, or pd.Series, of length n -- True if the the death was observed, False if the event
             was lost (right-censored). Defaults all True if event_observed==None
          timeline: iterable, optional
            return the best estimate at the values in timelines (postively increasing)
          label: string, optional
            a string to name the column of the estimate.
          alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
             alpha for this call to fit only.
          ci_labels: list, optional
            add custom column names to the generated confidence intervals
                as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>

        Returns
        -------
        self : ExpontentialFitter
          self, with new properties like 'survival_function_' and 'lambda_'.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )

        # check for negative or 0 durations - these are not allowed in a weibull model.
        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label


        (self.mu_, self.sigma_), self._log_likelihood = self._fit_model(self.durations, self.event_observed)
        

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(name=self._label)
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(name=self._label)
        self.median_ = np.exp(self.mu_)

        # estimation methods
        self._estimate_name = "cumulative_hazard_"
        self._predict_label = label
        self._update_docstrings()

        # plotting - Cumulative hazard takes priority.
        self.plot_cumulative_hazard = self.plot

        return self

    def _estimation_method(self, t):
        return self.survival_function_at_times(t)

    def hazard_at_times(self, times):        
        return pd.Series(norm.pdf((log(times) - self.mu_) / self.sigma_) / (self.sigma_ * times * self.survival_function_at_times(times)), index=_to_array(times))

    def survival_function_at_times(self, times):
        """
        Return a Pandas series of the predicted survival value at specific times

        Parameters
        -----------
        times: iterable or float

        Returns
        --------
        pd.Series

        """
        return pd.Series(1 - norm.cdf((log(times) - self.mu_) / self.sigma_), index=_to_array(times))

    def cumulative_hazard_at_times(self, times):
        timeline = np.linspace(0.001, np.max(times), 100)
        y = self.hazard_at_times(timeline)
        int_y = pd.DataFrame(cumtrapz(y, timeline, initial=0), index=timeline)
        return pd.Series(dataframe_interpolate_at_times(int_y, times).squeeze(), index=_to_array(times))

    def _fit_model(self, T, E):
        init = np.array([log(T).mean() + 1, log(T).std()])
        results = minimize(_negative_log_likelihood, init, 
                args=(T, E), 
                bounds=((None, None), (0.000001, None)),
                method='l-bfgs-b',
                options={'ftol': 2.220446049250313e-03}
        )
        if results.success:
            return results.x, -results.fun
        else:
            raise ConvergenceError("Did not converge.")

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, exp(coef), se(coef), z, p, lower, upper"""
        df = pd.DataFrame(index=["mu_", "sigma_"])
        df["coef"] = [self.mu_, self.sigma_]
        return df

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

        for k, v in kwargs.items():
            print("{} = {}\n".format(justify(k), v))

        print(end="\n")
        print("---")

        df = self.summary
        print(df.to_string(float_format=format_floats(decimals), formatters={"p": format_p_value(decimals)}))
