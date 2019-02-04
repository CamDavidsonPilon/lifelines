# -*- coding: utf-8 -*-
from __future__ import print_function, division
import autograd.numpy as np
from autograd import hessian, value_and_grad
from scipy.optimize import minimize

from scipy import stats
from numpy.linalg import inv
from lifelines.fitters import ParametericUnivariateFitter
from lifelines.utils import (
    _to_array,
    inv_normal_cdf,
    check_nans_or_infs,
    ConvergenceError,
    string_justify,
    format_p_value,
    format_floats,
)

class LogLogisticFitter(ParametericUnivariateFitter):

    r"""
    This class implements a Log-Logistic model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = (1 + (t/\alpha)^{\beta})^{-1},   \alpha > 0, \beta > 0,

    and the hazard rate is:

    .. math::  h(t) = (\beta/\alpha)(t / \alpha) ^ {\beta-1} / (1 + (t/\alpha)^{\beta})

    After calling the `.fit` method, you have access to properties like:
    ``cumulative_hazard_``, ``plot``, ``survival_function_``, ``alpha_`` and ``beta_``.

    A summary of the fit is available with the method 'print_summary()'
    
    Examples
    --------

    >>> from lifelines import LogLogisticFitter 
    >>> from lifelines.datasets import load_waltons
    >>> waltons = load_waltons()
    >>> llf = WeibullFitter()
    >>> llf.fit(waltons['T'], waltons['E'])
    >>> llf.plot()
    >>> print(llf.alpha_)

    """

    def fit(
        self,
        durations,
        event_observed=None,
        timeline=None,
        label="LogLogistic_estimate",
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
          self : LogLogisticFitter
            self with new properties like ``cumulative_hazard_``, ``survival_function_``, ``alpha_``, and ``beta_``.

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
        (self.alpha_, self.beta_), self._log_likelihood, self._hessian_ = self._fit_model(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self._fitted_parameters_ = np.array([self.alpha_, self.beta_])
        self._fitted_parameters_names_ = ["alpha_", "beta_"]

        self.variance_matrix_ = inv(self._hessian_)

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(name=self._label)
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(self._label)
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(self._label)

        self.confidence_interval_ = self._bounds(alpha, ci_labels)
        self.median_ = self.alpha_

        # estimation methods
        self._estimate_name = "cumulative_hazard_"
        self._predict_label = label
        self._update_docstrings()

        # plotting - Cumulative hazard takes priority.
        self.plot_cumulative_hazard = self.plot

        return self


    def _cumulative_hazard(self, params, times):
        alpha_, beta_ = params
        return np.log((times / alpha_) ** beta_ + 1)

