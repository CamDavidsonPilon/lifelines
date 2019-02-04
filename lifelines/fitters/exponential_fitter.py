# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats

from lifelines.fitters import UnivariateFitter
from lifelines.utils import _to_array, inv_normal_cdf, check_nans_or_infs, string_justify, format_p_value, format_floats
from lifelines.fitters import ParametericUnivariateFitter


class ExponentialFitter(ParametericUnivariateFitter):
    r"""
    This class implements an Exponential model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = exp(-(\lambda*t)),   \lambda >0

    which implies the cumulative hazard rate is

    .. math::  H(t) = \lambda*t

    and the hazard rate is:

    .. math::  h(t) = \lambda

    After calling the `.fit` method, you have access to properties like:
     'survival_function_', 'lambda_', 'cumulative_hazard_'

    A summary of the fit is available with the method 'print_summary()'

    Notes
    -----
    Reference: https://www4.stat.ncsu.edu/~dzhang2/st745/chap3.pdf

    """

    def fit(
        self, durations, event_observed=None, timeline=None, label="Exponential_estimate", alpha=None, ci_labels=None, show_progress=False
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
        self : ExponentialFitter
          self, with new properties like 'survival_function_', 'cumulative_hazard_', and 'lambda_'.

        """

        check_nans_or_infs(durations)
        if event_observed is not None:
            check_nans_or_infs(event_observed)

        self.durations = np.asarray(durations, dtype=float)
        self.event_observed = (
            np.asarray(event_observed, dtype=int) if event_observed is not None else np.ones_like(self.durations)
        )

        if timeline is not None:
            self.timeline = np.sort(np.asarray(timeline))
        else:
            self.timeline = np.linspace(self.durations.min(), self.durations.max(), self.durations.shape[0])

        self._label = label


        # estimation
        self.lambda_, self._log_likelihood, self._hessian_ = self._fit_model(
            self.durations, self.event_observed, show_progress=show_progress
        )
        self._fitted_parameters_ = np.array([self.lambda_])
        self._fitted_parameters_names_ = ["lambda_"]
        
        self.variance_matrix_ = 1. / self._hessian_

        self.survival_function_ = self.survival_function_at_times(self.timeline).to_frame(self._label)
        self.cumulative_hazard_ = self.cumulative_hazard_at_times(self.timeline).to_frame(self._label)
        self.hazard_ = self.hazard_at_times(self.timeline).to_frame(self._label)

        self.confidence_interval_ = self._bounds(alpha if alpha else self.alpha, ci_labels)
        self.median_ = 1.0 / self.lambda_ * (np.log(2))

        # estimation methods
        self._predict_label = label
        self._update_docstrings()

        return self

    def _fit_model(self, T, E, show_progress=True):
        lambda_ = E.sum() / T.sum()
        lambda_variance_ = lambda_ / T.sum()
        log_likelihood = np.log(lambda_) * E.sum() - lambda_ * T.sum()
        return lambda_, log_likelihood, np.array([[1. / lambda_variance_]])


    def _cumulative_hazard(self, params , times):
        lambda_ = params[0]
        return lambda_ * times

