# -*- coding: utf-8 -*-
import autograd.numpy as np
from lifelines.fitters import KnownModelParametricUnivariateFitter
from lifelines.utils.safe_exp import safe_exp
from lifelines import utils


class WeibullFitter(KnownModelParametricUnivariateFitter):

    r"""

    This class implements a Weibull model for univariate data. The model has parameterized
    form:

    .. math::  S(t) = \exp\left(-\left(\frac{t}{\lambda}\right)^\rho\right),   \lambda > 0, \rho > 0,

    The :math:`\lambda` (scale) parameter has an applicable interpretation: it represents the time when 63.2% of the population has died.
    The :math:`\rho` (shape) parameter controls if the cumulative hazard (see below) is convex or concave, representing accelerating or decelerating
    hazards.

    .. image:: /images/weibull_parameters.png

    The cumulative hazard rate is

    .. math:: H(t) = \left(\frac{t}{\lambda}\right)^\rho,

    and the hazard rate is:

    .. math::  h(t) = \frac{\rho}{\lambda}\left(\frac{t}{\lambda}\right)^{\rho-1}

    After calling the ``.fit`` method, you have access to properties like: ``cumulative_hazard_``, ``survival_function_``, ``lambda_`` and ``rho_``.
    A summary of the fit is available with the method ``print_summary()``.


    Parameters
    -----------
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Examples
    --------
    .. code:: python

        from lifelines import WeibullFitter
        from lifelines.datasets import load_waltons
        waltons = load_waltons()
        wbf = WeibullFitter()
        wbf.fit(waltons['T'], waltons['E'])
        wbf.plot()
        print(wbf.lambda_)

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
    lambda_: float
        The fitted parameter in the model
    rho_: float
        The fitted parameter in the model
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None

    Notes
    ----------
    Looking for a 3-parameter Weibull model? See notes `here <https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Piecewise%20Exponential%20Models%20and%20Creating%20Custom%20Models.html#3-parameter-Weibull-distribution>`_.
    """

    lambda_: float
    rho_: float
    _fitted_parameter_names = ["lambda_", "rho_"]
    _compare_to_values = np.array([1.0, 1.0])
    _scipy_fit_options = {"ftol": 1e-14}

    def _create_initial_point(self, Ts, E, entry, weights):
        return np.array([utils.coalesce(*Ts).mean(), 1.0])

    def _cumulative_hazard(self, params, times):
        lambda_, rho_ = params
        return safe_exp(rho_ * (np.log(np.clip(times, 1e-25, np.inf)) - np.log(lambda_)))

    def _log_hazard(self, params, times):
        lambda_, rho_ = params
        return np.log(rho_) - np.log(lambda_) + (rho_ - 1) * (np.log(times) - np.log(lambda_))

    def percentile(self, p) -> float:
        return self.lambda_ * (np.log(1.0 / p) ** (1.0 / self.rho_))
