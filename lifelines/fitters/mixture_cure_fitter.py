# -*- coding: utf-8 -*-
import autograd.numpy as anp
from lifelines.fitters import ParametricUnivariateFitter


class MixtureCureFitter(ParametricUnivariateFitter):
    CURED_FRACTION_PARAMETER_NAME = "cured_fraction_"

    def __init__(self, base_fitter, *args, **kwargs):
        self._base_fitter = base_fitter

        if self.CURED_FRACTION_PARAMETER_NAME in base_fitter._fitted_parameter_names:
            raise NameError(
                "'cured_fraction_' in _fitted_parameter_names is a lifelines reserved word. Try something "
                "else instead."
            )

        self._fitted_parameter_names = ["cured_fraction_"] + base_fitter._fitted_parameter_names
        self._bounds = [(0, 1)] + base_fitter._bounds
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
        return anp.array([0] + list(base_point))

    def percentile(self, p):
        c = self.cured_fraction_

        if p <= c:
            return anp.inf

        non_cure_p = (p - c) / (1 - c)
        return self._base_fitter.percentile(non_cure_p)
