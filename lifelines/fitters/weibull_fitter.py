# -*- coding: utf-8 -*-
from __future__ import print_function, division
import warnings
import numpy as np
import pandas as pd
import autograd.numpy as autograd_np
from autograd import elementwise_grad as egrad

from scipy import stats
from numpy.linalg import inv
from lifelines.fitters import ParametericUnivariateFitter
from lifelines.utils import (
    _to_array,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
)


class WeibullFitter(ParametericUnivariateFitter):

    r"""
    This class implements a Weibull model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = exp(-(\lambda t)^\rho),   \lambda > 0, \rho > 0,

    which implies the cumulative hazard rate is

    .. math:: H(t) = (\lambda t)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \rho \lambda(\lambda t)^{\rho-1}

    After calling the `.fit` method, you have access to properties like:
    ``cumulative_hazard_``, ``survival_function_``, ``lambda_`` and ``rho_``.

    A summary of the fit is available with the method 'print_summary()'
    
    Examples
    --------

    >>> from lifelines import WeibullFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> wbf = WeibullFitter()
    >>> wbf.fit(waltons['T'], waltons['E'])
    >>> wbf.plot()
    >>> print(wbf.lambda_)

    """

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label="Weibull_estimate",
        alpha=None,
        ci_labels=None,
        show_progress=False,
    ):  # pylint: disable=too-many-arguments
        """
        Parameters
        ----------
        durations: an array, or pd.Series
          length n, duration subject was observed for
        event_observed: numpy array or pd.Series, optional
          length n, True if the the death was observed, False if the event
           was lost (right-censored). Defaults all True if event_observed==None
        timeline: list, optional
            return the estimate at the values in timeline (postively increasing)
        label: string, optional
            a string to name the column of the estimate.
        alpha: float, optional
            the alpha value in the confidence intervals. Overrides the initializing
           alpha for this call to fit only.
        ci_labels: list, optional
            add custom column names to the generated confidence intervals
              as a length-2 list: [<lower-bound name>, <upper-bound name>]. Default: <label>_lower_<alpha>
        show_progress: boolean, optional
            since this is an iterative fitting algorithm, switching this to True will display some iteration details.

        Returns
        -------
          self : WeibullFitter
            self with new properties like ``cumulative_hazard_``, ``survival_function_``, ``lambda_``, and ``rho_``.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        # check for negative or 0 durations - these are not allowed in a weibull model.
        if np.any(self.durations <= 0):
            raise ValueError(
                "This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements."
            )

        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label
        alpha = alpha if alpha is not None else self.alpha

        # estimation
        (self.lambda_, self.rho_), self._log_likelihood, self._hessian_ = self._fit_model(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self._fitted_parameters_ = np.array([self.lambda_, self.rho_])
        self._fitted_parameters_names_ = ["lambda_", "rho_"]
        
        self.variance_matrix_ = inv(self._hessian_)

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(self._label)
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(self._label)
       
        self.confidence_interval_ = self._bounds(alpha, ci_labels)
        
        self.median_ = 1.0 / self.lambda_ * (np.log(2)) ** (1.0 / self.rho_)

        # estimation methods
        self._estimate_name = "cumulative_hazard_"
        self._predict_label = label
        self._update_docstrings()


        # plotting - Cumulative hazard takes priority.
        self.plot_cumulative_hazard = self.plot
    
        return self

    def _cumulative_hazard(self, params, times):
        lambda_, rho_ = params
        return (lambda_ * times) ** rho_

