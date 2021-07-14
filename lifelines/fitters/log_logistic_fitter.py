# -*- coding: utf-8 -*-
import autograd.numpy as np

from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils import CensoringType


class LogLogisticFitter(KnownModelParametricUnivariateFitter):

    r"""
    This class implements a Log-Logistic model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = \left(1 + \left(\frac{t}{\alpha}\right)^{\beta}\right)^{-1},   \alpha > 0, \beta > 0,

    The :math:`\alpha` (scale) parameter has an interpretation as being equal to the *median* lifetime of the population. The
    :math:`\beta` parameter influences the shape of the hazard. See figure below:

    .. image:: /images/log_normal_alpha.png

    The hazard rate is:

    .. math::  h(t) = \frac{\left(\frac{\beta}{\alpha}\right)\left(\frac{t}{\alpha}\right) ^ {\beta-1}}{\left(1 + \left(\frac{t}{\alpha}\right)^{\beta}\right)}

    and the cumulative hazard is:

    .. math:: H(t) = \log\left(\left(\frac{t}{\alpha}\right) ^ {\beta} + 1\right)

    After calling the ``.fit`` method, you have access to properties like: ``cumulative_hazard_``, ``plot``, ``survival_function_``, ``alpha_`` and ``beta_``.
    A summary of the fit is available with the method 'print_summary()'

    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Examples
    --------
    .. code:: python

        from lifelines import LogLogisticFitter
        from lifelines.datasets import load_waltons
        waltons = load_waltons()

        llf = LogLogisticFitter()
        llf.fit(waltons['T'], waltons['E'])
        llf.plot()
        print(llf.alpha_)

    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    cumulative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    density_: DataFrame
        The estimated density function (PDF) (with custom timeline if provided)
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    median_survival_time_: float
        The median time to event
    alpha_: float
        The fitted parameter in the model
    beta_: float
        The fitted parameter in the model
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    """

    alpha_: float
    beta_: float
    _fitted_parameter_names = ["alpha_", "beta_"]
    _compare_to_values = np.array([1.0, 1.0])

    def percentile(self, p):
        a = self.alpha_
        b = self.beta_
        return a * (1 / (1 - p) - 1) ** (-1 / b)

    def _create_initial_point(self, Ts, E, *args):
        if CensoringType.is_right_censoring(self):
            T = Ts[0]
        elif CensoringType.is_left_censoring(self):
            T = np.clip(0.0001, np.inf, Ts[1])
        elif CensoringType.is_interval_censoring(self):
            if E.sum() > 0:
                # Ts[1] can contain infs, so ignore this data
                okay_data = Ts[1] < 1e10
                T = Ts[1]
                T = T[okay_data]
            else:
                T = np.array([1.0])
        return np.array([np.median(T), 1.0])

    @property
    def median_survival_time_(self):
        return self.alpha_

    def _cumulative_hazard(self, params, times):
        alpha_, beta_ = params
        return np.logaddexp(beta_ * (np.log(np.clip(times, 1e-25, np.inf)) - np.log(alpha_)), 0)

    def _log_1m_sf(self, params, times):
        alpha_, beta_ = params
        return -np.logaddexp(-beta_ * (np.log(times) - np.log(alpha_)), 0)
