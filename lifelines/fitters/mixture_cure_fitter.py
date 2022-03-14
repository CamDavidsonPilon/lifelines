# -*- coding: utf-8 -*-
import autograd.numpy as anp
from lifelines.fitters import ParametricUnivariateFitter


class MixtureCureFitter(ParametricUnivariateFitter):
    r"""

    This class implements a Mixture Cure Model for univariate data with a configurable distribution for the
    non-cure portion. The model survival function has parameterized form:

    .. math::  S(t) = c + \left(1 - c\right)S_b(t),  \;\; 1 > c > 0

    where :math:`S_b(t)` is a parametric survival function describing the non-cure portion of the population, and
    :math:`c` is the cured fraction of the population.

    After calling the ``.fit`` method, you have access to properties like: ``cumulative_hazard_``,
    ``survival_function_``, ``lambda_`` and ``rho_``. A summary of the fit is available with the method
    ``print_summary()``. The parameters for both the cure portion of the model and from the base_fitter are available.
    The cure fraction is called ``cured_fraction_``, and parameters from the base_fitter will be available with
    their own appropriate names.

    Parameters
    -----------
    base_fitter: ParametricUnivariateFitter, required
        an instance of a fitter that describes the non-cure portion of the population.
    alpha: float, optional (default=0.05)
        the level in the confidence intervals.

    Important
    ----------
    The **base_fitter** instance is used to describe the non-cure portion of the population, but is not actually
    fit to the data. Some internal properties are modified, and it should not be used for any other purpose after
    passing it to the constructor of this class.

    Examples
    --------

    .. code:: python

        from lifelines import MixtureCureFitter, ExponentialFitter

        fitter = MixtureCureFitter(base_fitter=ExponentialFitter())
        fitter.fit(T, event_observed=observed)
        print(fitter.cured_fraction_)
        print(fitter.lambda_)  # This is available because it is a parameter of the ExponentialFitter

    Attributes
    ----------
    cumulative_hazard_ : DataFrame
        The estimated cumulative hazard (with custom timeline if provided)
    cured_fraction_ : float
        The fitted parameter :math:`c` in the model
    hazard_ : DataFrame
        The estimated hazard (with custom timeline if provided)
    survival_function_ : DataFrame
        The estimated survival function (with custom timeline if provided)
    cumulative_density_ : DataFrame
        The estimated cumulative density function (with custom timeline if provided)
    variance_matrix_ : DataFrame
        The variance matrix of the coefficients
    median_survival_time_: float
        The median time to event
    durations: array
        The durations provided
    event_observed: array
        The event_observed variable provided
    timeline: array
        The time line to use for plotting and indexing
    entry: array or None
        The entry array provided, or None
    """

    _KNOWN_MODEL = True
    _CURED_FRACTION_PARAMETER_NAME = "cured_fraction_"

    def __init__(self, base_fitter, *args, **kwargs):
        self._base_fitter = base_fitter

        if self._CURED_FRACTION_PARAMETER_NAME in base_fitter._fitted_parameter_names:
            raise NameError(
                f"'{self._CURED_FRACTION_PARAMETER_NAME}' in _fitted_parameter_names is a lifelines reserved word."
                f" Try something else instead."
            )

        self._compare_to_values = anp.append([0.0], self._base_fitter._compare_to_values)
        self._fitted_parameter_names = [self._CURED_FRACTION_PARAMETER_NAME] + base_fitter._fitted_parameter_names
        self._bounds = [(0, 1)] + base_fitter._bounds
        self._scipy_fit_options = base_fitter._scipy_fit_options
        super().__init__(*args, **kwargs)

    def _fit(self, *args, **kwargs):
        self._base_fitter._censoring_type = self._censoring_type
        result = super()._fit(*args, **kwargs)
        for param_name, fitted_value in zip(self._fitted_parameter_names[1:], self._fitted_parameters_[1:]):
            setattr(self._base_fitter, param_name, fitted_value)
        return result

    def _cumulative_hazard(self, params, times):
        c = params[0]
        base_survival_function = self._base_fitter._survival_function(params[1:], times)
        return -anp.log(c + (1 - c) * base_survival_function)

    def _survival_function(self, params, times):
        c = params[0]
        base_cumulative_hazard = self._base_fitter._cumulative_hazard(params[1:], times)
        return c + (1 - c) * (anp.exp(-base_cumulative_hazard))

    def _log_hazard(self, params, times):
        c = params[0]
        base_survival_function = self._base_fitter._survival_function(params[1:], times)
        base_log_hazard = self._base_fitter._log_hazard(params[1:], times)
        cumulative_hazard = -anp.log(c + (1 - c) * base_survival_function)
        return anp.log(1 - c) + anp.log(base_survival_function) + base_log_hazard + cumulative_hazard

    def _create_initial_point(self, Ts, E, *args):
        base_point = self._base_fitter._create_initial_point(Ts, E, *args)
        return anp.array([0.5] + list(base_point))

    def percentile(self, p):
        c = getattr(self, self._CURED_FRACTION_PARAMETER_NAME)

        if p <= c:
            return anp.inf

        non_cure_p = (p - c) / (1 - c)
        return self._base_fitter.percentile(non_cure_p)
